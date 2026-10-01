import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from assignment1_common import CONFIG, DEVICE, train_loader


class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)


def train_gan(generator, discriminator):
    g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    history = []
    started = __import__("time").perf_counter()
    for epoch in range(CONFIG["gan_epochs"]):
        generator.train()
        discriminator.train()
        g_loss_total = 0.0
        d_loss_total = 0.0
        for images, _ in tqdm(train_loader, desc=f"GAN {epoch + 1}", leave=False):
            images = images.to(DEVICE)
            batch = images.size(0)
            real_label = torch.ones(batch, device=DEVICE)
            fake_label = torch.zeros(batch, device=DEVICE)

            d_opt.zero_grad(set_to_none=True)
            real_logits = discriminator(images).view(-1)
            real_loss = F.binary_cross_entropy_with_logits(real_logits, real_label)
            noise = torch.randn(batch, generator.latent_dim, device=DEVICE)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach()).view(-1)
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_label)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad(set_to_none=True)
            noise = torch.randn(batch, generator.latent_dim, device=DEVICE)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images).view(-1)
            g_loss = F.binary_cross_entropy_with_logits(fake_logits, real_label)
            g_loss.backward()
            g_opt.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
        history.append({"epoch": epoch + 1, "g_loss": g_loss_total / len(train_loader), "d_loss": d_loss_total / len(train_loader)})
    return generator, discriminator, history, __import__("time").perf_counter() - started


@torch.no_grad()
def sample_gan(generator, count):
    noise = torch.randn(count, generator.latent_dim, device=DEVICE)
    return generator(noise).clamp(-1, 1)
