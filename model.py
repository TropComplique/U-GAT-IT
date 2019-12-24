import cv2
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from networks import Generator
from networks import RhoClipper
from networks import Discriminator
from input_pipeline import Images


class UGATIT:

    def __init__(self):

        batch_size = 1
        image_size = 256
        device = torch.device('cuda:0')
        data = '/home/dan/datasets/selfie2anime/'

        self.model_save_prefix = 'models/run00'
        logs_dir = 'summaries/run00/'
        self.writer = SummaryWriter(logs_dir)

        num_steps = 1000000
        self.num_steps = num_steps
        self.save_step = 100000
        self.plot_image_step = 3000
        self.plot_loss_step = 30

        size = (image_size, image_size)
        self.dataset = {
            'train_A': Images(os.path.join(data, 'train_A'), size, is_training=True),
            'train_B': Images(os.path.join(data, 'train_B'), size, is_training=True),
            # 'test_A': Images(os.path.join(data, 'test_A'), size, is_training=False),
            # 'test_B': Images(os.path.join(data, 'test_B'), size, is_training=False)
        }

        def get_loader(dataset, is_training):
            return DataLoader(
                dataset=dataset, shuffle=is_training,
                batch_size=batch_size if is_training else 1,
                num_workers=1, pin_memory=True, drop_last=True
            )

        self.loader = {
            k: get_loader(v, v.is_training)
            for k, v in self.dataset.items()
        }

        generator = {
            'A2B': Generator(),
            'B2A': Generator()
        }

        discriminator = {
            'global_A': Discriminator(downsample=5),
            'global_B': Discriminator(downsample=5),
            'local_A': Discriminator(downsample=3),
            'local_B': Discriminator(downsample=3)
        }

        self.generator = nn.ModuleDict(generator)
        self.discriminators = nn.ModuleDict(discriminators)

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                init.ones_(m.weight)
                init.zeros_(m.bias)

        self.generator.apply(weights_init).to(device)
        self.discriminator.apply(weights_init).to(device)

        params = {
            'lr': 1e-4,
            'betas': (0.5, 0.999),
            'weight_decay': 1e-4
        }

        self.G_optimizer = optim.Adam(self.generator.parameters(), **params)
        self.D_optimizer = optim.Adam(self.discriminators.parameters(), **params)

        def lambda_rule(i):
            decay = int(0.5 * num_steps)
            m = 1.0 if i < decay else 1.0 - (i - decay) / (num_steps - decay)
            return max(m, 1e-3)

        self.schedulers = [
            LambdaLR(self.G_optimizer, lr_lambda=lambda_rule),
            LambdaLR(self.D_optimizer, lr_lambda=lambda_rule)
        ]

        self.rho_clipper = RhoClipper()

    def get_discriminator_losses(self, real, fake, domain):
        """
        Arguments:
            real: a float tensor with shape [n, c, h, w].
            fake: a float tensor with shape [n, c, h, w].
            domain: a string.
        Returns:
            a dict that contains float tensors with shape [].
        """

        losses = {}

        def mse_loss(x, y):
            return (x - 1.0).pow(2).mean() + y.pow(2).mean()

        for scale in ['local', 'global']:

            network = self.discriminator[f'{scale}_{domain}']

            real_score, real_cam_logit, _ = network(real)
            fake_score, fake_cam_logit, _ = network(fake)

            losses[f'{domain}_{scale}'] = mse_loss(real_score, fake_score)
            losses[f'{domain}_{scale}_cam'] = mse_loss(real_cam_logit, fake_cam_logit)

        return losses

    def discriminators_step(self, real_A, real_B, fake_A2B, fake_B2A):
        """
        Arguments:
            real_A, fake_B2A: float tensors with shape [n, a, h, w].
            real_B, fake_A2B: float tensors with shape [n, b, h, w].
        Returns:
            a dict with float numbers.
        """

        self.D_optimizer.zero_grad()

        fake_A2B = fake_A2B.detach()
        fake_B2A = fake_B2A.detach()

        losses = {}
        losses.update(self.get_discriminator_losses(real=real_A, fake=fake_B2A, domain='A'))
        losses.update(self.get_discriminator_losses(real=real_B, fake=fake_A2B, domain='B'))

        discriminator_loss = sum(x for x in losses.values())
        discriminator_loss.backward()
        self.D_optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def generators_step(self, real_A, real_B, fake_A2B, fake_B2A, fake_A2B_cam_logit, fake_B2A_cam_logit):

        self.discriminators.requires_grad_(False)
        self.G_optimizer.zero_grad()

        fake_A2B2A, _, _ = self.generator['B2A'](fake_A2B)
        fake_B2A2B, _, _ = self.generator['A2B'](fake_B2A)

        def get_discriminator_losses(fake, domain):

            def mse_loss(x):
                return (x - 1.0).pow(2).mean()

            for scale in ['local', 'global']:

                network = self.discriminator[f'{scale}_{domain}']
                fake_score, fake_cam_logit, _ = network(fake)

                losses[f'g_{domain}_{scale}'] = mse_loss(fake_score)
                losses[f'g_{domain}_{scale}_cam'] = mse_loss(fake_cam_logit)

        losses = {}
        losses.update(get_discriminator_losses(fake=fake_B2A, domain='A'))
        losses.update(get_discriminator_losses(fake=fake_A2B, domain='B'))

        losses['reconstruction_A'] = 10.0 * F.l1_loss(fake_A2B2A, real_A)
        losses['reconstruction_B'] = 10.0 * F.l1_loss(fake_B2A2B, real_B)

        fake_A2A, fake_A2A_cam_logit, _ = self.generator['B2A'](real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.generator['A2B'](real_B)

        losses['identity_A'] = 10.0 * F.l1_loss(fake_A2A, real_A)
        losses['identity_B'] = 10.0 * F.l1_loss(fake_B2B, real_B)

        def bce(x, is_true):
            target = torch.ones_like(x) if is_true else torch.zeros_like(x)
            return F.binary_cross_entropy_with_logits(x, target)

        losses['cam_A'] = 1000.0 * (bce(fake_B2A_cam_logit, True) + bce(fake_A2A_cam_logit, False))
        losses['cam_B'] = 1000.0 * (bce(fake_A2B_cam_logit, True) + bce(fake_B2B_cam_logit, False))

        generator_loss = sum(x for x in losses.values())
        generator_loss.backward()
        self.G_optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def train(self):

        for step in range(self.num_steps):

            print(f'iteration {step}')

            try:
                real_A = train_A_iterator.next()
            except:
                train_A_iterator = iter(self.loader['train_A'])
                real_A = train_A_iterator.next()

            try:
                real_B = train_B_iterator.next()
            except:
                train_B_iterator = iter(self.loader['train_B'])
                real_B = train_B_iterator.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            fake_A2B, fake_A2B_cam_logit, _ = self.generator['A2B'](real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.generator['B2A'](real_B)

            losses = {}
            losses.update(self.discriminators_step(real_A, real_B, fake_A2B, fake_B2A))
            losses.update(self.generators_step(real_A, real_B, fake_A2B, fake_B2A, fake_A2B_cam_logit, fake_B2A_cam_logit))

            if step % self.plot_loss_step == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(k, v, step)

            if step % self.save_step == 0:
                self.save()

            self.generator.apply(self.rho_clipper)

            for s in self.schedulers:
                s.step()

            if step % self.plot_image_step == 0:
                A2B, B2A = self.visualize(real_A, real_B)
                self.writer.add_image(f'result_A2B', A2B, step)
                self.writer.add_image(f'result_B2A', B2A, step)

    def visualize(self, real_A, real_B):
        """
        Arguments:
            real_A: a float tensor with shape [1, 3, h, w].
            real_B: a float tensor with shape [1, 3, h, w].
        Returns:
            A2B: a float tensor with shape [1, 3, h, 7 * w].
            B2A: a float tensor with shape [1, 3, h, 7 * w].
        """

        with torch.no_grad():

            fake_A2B, _, fake_A2B_heatmap = self.generator['A2B'](real_A)
            fake_B2A, _, fake_B2A_heatmap = self.generator['B2A'](real_B)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.generator['B2A'](fake_A2B)
            fake_B2A2B, _, fake_B2A2B_heatmap = self.generator['A2B'](fake_B2A)

            fake_A2A, _, fake_A2A_heatmap = self.generator['B2A'](real_A)
            fake_B2B, _, fake_B2B_heatmap = self.generator['A2B'](real_B)

        def visualize_cam(x, size=256):
            """
            Arguments:
                x: a float tensor with shape [1, 1, h / s, w / s].
            Returns:
                a float tensor with shape [3, h, w].
            """

            x = x.squeeze(0).cpu()
            x = x.permute(1, 2, 0).numpy()
            # it has shape [h / s, w / s, 1]

            x = x - np.min(x)
            x = x / np.max(x)

            x = np.uint8(255.0 * x)
            x = cv2.resize(x, (size, size))
            x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
            x = np.float32(x) / 255.0

            x = torch.FloatTensor(x)
            return x.permute(2, 0, 1)

        A2B = torch.cat([
            real_A[0].cpu(),
            visualize_cam(fake_A2A_heatmap), fake_A2A[0].cpu(),
            visualize_cam(fake_A2B_heatmap), fake_A2B[0].cpu(),
            visualize_cam(fake_A2B2A_heatmap), fake_A2B2A[0].cpu()
        ], dim=2)

        B2A = torch.cat([
            real_B[0].cpu(),
            visualize_cam(fake_B2B_heatmap), fake_B2B[0].cpu(),
            visualize_cam(fake_B2A_heatmap), fake_B2A[0].cpu(),
            visualize_cam(fake_B2A2B_heatmap), fake_B2A2B[0].cpu()
        ], dim=2)

        return A2B, B2A

    def save_model(self):

        torch.save(self.generator.state_dict(), f'{self.model_save_prefix}_generator.pth')
        torch.save(self.discriminator.state_dict(), f'{self.model_save_prefix}_discriminator.pth')
