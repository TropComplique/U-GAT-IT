import os
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


# this will give a speed up
torch.backends.cudnn.benchmark = True


class UGATIT:

    def __init__(self):

        batch_size = 1
        num_steps = 500000
        image_size = 256
        device = torch.device('cuda:0')

        # download the data from here:
        # https://github.com/taki0112/UGATIT#dataset

        train_A_path = '/home/dan/datasets/selfie2anime/trainA/'
        train_B_path = '/home/dan/datasets/selfie2anime/trainB/'

        save_dir = 'models/'
        name = 'run00'  # model name
        logs_dir = 'summaries/run00/'

        # use this to restore training
        self.start_step = None

        self.name = name
        self.save_dir = save_dir
        self.writer = SummaryWriter(logs_dir)

        self.device = device
        self.num_steps = num_steps
        self.save_step = 50000
        self.plot_image_step = 3000
        self.plot_loss_step = 10

        size = (image_size, image_size)
        self.dataset = {
            'train_A': Images(train_A_path, size),
            'train_B': Images(train_B_path, size)
        }

        def get_loader(dataset):
            return DataLoader(
                dataset=dataset, shuffle=True,
                batch_size=batch_size, num_workers=1,
                pin_memory=True, drop_last=True
            )

        self.loader = {k: get_loader(v) for k, v in self.dataset.items()}

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
        self.discriminator = nn.ModuleDict(discriminator)

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                init.ones_(m.weight)
                init.zeros_(m.bias)

        self.generator.apply(weights_init).to(device).train()
        self.discriminator.apply(weights_init).to(device).train()

        params = {
            'lr': 2e-4,
            'betas': (0.5, 0.999),
            'weight_decay': 0.0
        }

        self.G_optimizer = optim.Adam(self.generator.parameters(), **params)
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), **params)

        def lambda_rule(i):
            decay = int(0.5 * num_steps)
            m = 1.0 if i < decay else 1.0 - (i - decay) / (num_steps - decay)
            return max(m, 1e-3)

        self.G_scheduler = LambdaLR(self.G_optimizer, lr_lambda=lambda_rule)
        self.D_scheduler = LambdaLR(self.D_optimizer, lr_lambda=lambda_rule)
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

        return {f'discriminators/{k}': v.item() for k, v in losses.items()}

    def generators_step(self, real_A, real_B, fake_A2B, fake_B2A, fake_A2B_cam_logit, fake_B2A_cam_logit):

        self.discriminator.requires_grad_(False)
        self.G_optimizer.zero_grad()

        fake_A2B2A, _, _ = self.generator['B2A'](fake_A2B)
        fake_B2A2B, _, _ = self.generator['A2B'](fake_B2A)

        def get_discriminator_losses(fake, domain):

            losses = {}

            def mse_loss(x):
                return (x - 1.0).pow(2).mean()

            for scale in ['local', 'global']:

                network = self.discriminator[f'{scale}_{domain}']
                fake_score, fake_cam_logit, _ = network(fake)

                losses[f'{domain}_{scale}'] = mse_loss(fake_score)
                losses[f'{domain}_{scale}_cam'] = mse_loss(fake_cam_logit)

            return losses

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
        self.discriminator.requires_grad_(True)

        return {f'generators/{k}': v.item() for k, v in losses.items()}

    def train(self):

        if self.start_step is not None:
            start_step = self.start_step + 1
            self.load(self.start_step)
        else:
            start_step = 1

        for step in range(start_step, self.num_steps + 1):

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

            self.generator.apply(self.rho_clipper)
            self.G_scheduler.step()
            self.D_scheduler.step()

            if step % self.save_step == 0:
                self.save(step)

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

    def save(self, step):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        name = f'{self.name}_step_{step}'
        path = os.path.join(self.save_dir, name)

        torch.save(self.generator.state_dict(), f'{path}_generators.pth')
        torch.save(self.discriminator.state_dict(), f'{path}_discriminators.pth')

        training_state = {
            'G_optimizer': self.G_optimizer.state_dict(),
            'D_optimizer': self.D_optimizer.state_dict(),
            'G_scheduler': self.G_scheduler.state_dict(),
            'D_scheduler': self.D_scheduler.state_dict()
        }
        torch.save(training_state, f'{path}_training_state.pth')

    def load(self, step):

        name = f'{self.name}_step_{step}'
        path = os.path.join(self.save_dir, name)

        self.generator.load_state_dict(torch.load(f'{path}_generators.pth'))
        self.discriminator.load_state_dict(torch.load(f'{path}_discriminators.pth'))

        training_state = torch.load(f'{path}_training_state.pth')
        self.G_optimizer.load_state_dict(training_state['G_optimizer'])
        self.D_optimizer.load_state_dict(training_state['D_optimizer'])
        self.G_scheduler.load_state_dict(training_state['G_scheduler'])
        self.D_scheduler.load_state_dict(training_state['D_scheduler'])
