import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, depth=64, downsample=2, num_blocks=4, image_size=256):
        super(Generator, self).__init__()

        down_path = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, depth, kernel_size=7, bias=False),
            nn.InstanceNorm2d(depth, affine=True),
            nn.ReLU(inplace=True)
        ]

        for i in range(downsample):
            m = 2**i  # multiplier
            down_path.extend([
                nn.ReflectionPad2d(1),
                nn.Conv2d(depth * m, depth * m * 2, kernel_size=3, stride=2, bias=False),
                nn.InstanceNorm2d(depth * m * 2, affine=True),
                nn.ReLU(inplace=True)
            ])

        m = 2**downsample
        for i in range(num_blocks):
            down_path.append(ResnetBlock(depth * m))

        self.gap_fc = nn.Linear(depth * m, 1, bias=False)
        self.gmp_fc = nn.Linear(depth * m, 1, bias=False)
        self.conv1x1 = nn.Conv2d(depth * m * 2, depth * m, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        def downsampling_block(d):
            return nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(d, d, kernel_size=3, stride=2, bias=False),
                nn.InstanceNorm2d(d, affine=True),
                nn.ReLU(inplace=True)
            )

        # downsampled spatial size
        size = image_size // m

        num_additional_downsamplings = 2
        size //= 2**num_additional_downsamplings

        FC = [
            downsampling_block(depth * m),
            downsampling_block(depth * m),
            downsampling_block(depth * m),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(depth * m, depth * m),
            nn.ReLU(inplace=True),
            nn.Linear(depth * m, depth * m),
            nn.ReLU(inplace=True)
        ]

        self.gamma = nn.Linear(depth * m, depth * m)
        self.beta = nn.Linear(depth * m, depth * m)

        styled_blocks = []
        for i in range(num_blocks):
            styled_blocks.append(ResnetAdaILNBlock(depth * m))

        up_path = []
        for i in range(downsample):
            m = 2**(downsample - i)
            up_path.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(depth * m, depth * m // 2, kernel_size=3),
                ILN(depth * m // 2),
                nn.ReLU(inplace=True)
            ])

        up_path.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(depth, out_channels, kernel_size=7),
            nn.Tanh()
        ])

        self.down_path = nn.Sequential(*down_path)
        self.FC = nn.Sequential(*FC)
        self.styled_blocks = nn.ModuleList(styled_blocks)
        self.up_path = nn.Sequential(*up_path)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            x: a float tensor with shape [b, out_channels, h, w].
            cam_logit: a float tensor with shape [b, 2].
            heatmap: a float tensor with shape [b, 1, h / s, w / s],
                where s = 2**downsample.
        """

        # batch size
        b = x.shape[0]

        x = 2.0 * x - 1.0
        x = self.down_path(x)
        # it has shape [b, depth * s, h / s, w / s],
        # where s = 2**downsample

        gap = F.adaptive_avg_pool2d(x, 1)
        # it has shape [b, depth * s, 1, 1]

        gap_logit = self.gap_fc(gap.view(b, -1))
        # it has shape [b, 1]

        gap_weight = self.gap_fc.weight
        # it has shape [1, depth * s]

        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        # it has shape [b, depth * s, h / s, w / s]

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(b, -1))
        gmp_weight = self.gmp_fc.weight
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        # it has shape [b, depth * s, h / s, w / s]

        cam_logit = torch.cat([gap_logit, gmp_logit], dim=1)
        # it has shape [b, 2]

        x = torch.cat([gap, gmp], dim=1)
        x = self.relu(self.conv1x1(x))
        # it has shape [b, depth * s, h / s, w / s]

        heatmap = torch.sum(x, dim=1, keepdim=True)
        # it has shape [b, 1, h / s, w / s]

        y = self.FC(x)
        gamma = self.gamma(y)
        beta = self.beta(y)
        # they have shape [b, depth * s]

        for m in self.styled_blocks:
            x = m(x, gamma, beta)
            # it has shape [b, depth * s, h / s, w / s]

        x = self.up_path(x)
        x = 0.5 * x + 0.5
        return x, cam_logit, heatmap


class ResnetBlock(nn.Module):

    def __init__(self, depth):
        super(ResnetBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(depth, depth, kernel_size=3, bias=False),
            nn.InstanceNorm2d(depth, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(depth, depth, kernel_size=3, bias=False),
            nn.InstanceNorm2d(depth, affine=True)
        )

    def forward(self, x):
        return x + self.layers(x)


class ResnetAdaILNBlock(nn.Module):

    def __init__(self, depth):
        super(ResnetAdaILNBlock, self).__init__()

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(depth, depth, kernel_size=3)
        self.norm1 = adaILN(depth)
        self.relu1 = nn.ReLU(inplace=True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3)
        self.norm2 = adaILN(depth)

    def forward(self, x, gamma, beta):

        y = self.pad1(x)
        y = self.conv1(y)
        y = self.norm1(y, gamma, beta)
        y = self.relu1(y)
        y = self.pad2(y)
        y = self.conv2(y)
        y = self.norm2(y, gamma, beta)

        return y + x


class adaILN(nn.Module):

    def __init__(self, num_features):
        super(adaILN, self).__init__()

        rho = torch.FloatTensor(1, num_features, 1, 1)
        self.rho = nn.Parameter(rho.fill_(0.9))

    def forward(self, x, gamma, beta):

        epsilon = 1e-3

        in_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        in_var = torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + epsilon)

        ln_mean = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        ln_var = torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + epsilon)

        x = self.rho * out_in + (1.0 - self.rho) * out_ln
        x = x * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return x


class ILN(nn.Module):

    def __init__(self, num_features):
        super(ILN, self).__init__()

        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, x):

        epsilon = 1e-3

        in_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        in_var = torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + epsilon)

        ln_mean = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        ln_var = torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + epsilon)

        x = self.rho * out_in + (1.0 - self.rho) * out_ln
        x = x * self.gamma + self.beta

        return x


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, depth=64, downsample=3):
        super(Discriminator, self).__init__()

        from torch.nn.utils import spectral_norm

        model = [
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels, depth, kernel_size=4, stride=2)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(1, downsample):
            m = 2 ** (i - 1)
            model.extend([
                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(depth * m, depth * m * 2, kernel_size=4, stride=2)),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        m = 2 ** (downsample - 1)
        model.extend([
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(depth * m, depth * m * 2, kernel_size=4)),
            nn.LeakyReLU(0.2, True)
        ])

        m = 2 ** downsample
        self.gap_fc = spectral_norm(nn.Linear(depth * m, 1, bias=False))
        self.gmp_fc = spectral_norm(nn.Linear(depth * m, 1, bias=False))
        self.conv1x1 = nn.Conv2d(depth * m * 2, depth * m, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.model = nn.Sequential(*model)
        self.pad = nn.ReflectionPad2d(1)
        self.conv = spectral_norm(nn.Conv2d(depth * m, 1, kernel_size=4))

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            x: a float tensor with shape [b, 1, h / s, w / s].
            cam_logit: a float tensor with shape [b, 2].
            heatmap: a float tensor with shape [b, 1, h / s, w / s],
                where s = 2**downsample.
        """
        b = x.shape[0]

        x = 2.0 * x - 1.0
        x = self.model(x)
        # it has shape [b, depth * s, h / s, w / s],
        # where s = 2**downsample

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(b, -1))
        gap_weight = self.gap_fc.weight
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(b, -1))
        gmp_weight = self.gmp_fc.weight
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], dim=1)
        # it has shape [b, 2]

        x = torch.cat([gap, gmp], dim=1)
        x = self.leaky_relu(self.conv1x1(x))
        # it has shape [b, depth * s, h / s, w / s]

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        x = self.conv(x)

        return x, cam_logit, heatmap


class RhoClipper:

    def __init__(self):
        pass

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(0.0, 1.0)
            module.rho.data = w
