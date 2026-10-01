import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from assignment1_common import CONFIG, DEVICE, train_loader


class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(128 * 2 * 2, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 128, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def train_vae(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    history = []
    started = __import__("time").perf_counter()
    for epoch in range(CONFIG["vae_epochs"]):
        model.train()
        recon_total = 0.0
        kl_total = 0.0
        loss_total = 0.0
        for images, _ in tqdm(train_loader, desc=f"VAE {epoch + 1}", leave=False):
            images = images.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(images)
            recon_loss = F.mse_loss(recon, images, reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.1 * kl
            loss.backward()
            optimizer.step()
            recon_total += recon_loss.item()
            kl_total += kl.item()
            loss_total += loss.item()
        history.append({
            "epoch": epoch + 1,
            "recon_loss": recon_total / len(train_loader),
            "kl": kl_total / len(train_loader),
            "total_loss": loss_total / len(train_loader),
        })
    return model, history, __import__("time").perf_counter() - started


@torch.no_grad()
def sample_vae(model, count):
    z = torch.randn(count, model.latent_dim, device=DEVICE)
    return model.decode(z).clamp(-1, 1)
