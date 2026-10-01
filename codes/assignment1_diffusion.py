import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from assignment1_common import CONFIG, DEVICE, train_loader


class TinyUNet(nn.Module):
    def __init__(self, base_channels=32, time_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.inc = nn.Conv2d(1, base_channels, 3, padding=1)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x, timestep):
        timestep = timestep.to(device=x.device, dtype=x.dtype)
        frequencies = torch.arange(32, device=x.device, dtype=x.dtype)
        frequencies = torch.exp(-torch.log(torch.tensor(10000.0, device=x.device, dtype=x.dtype)) * frequencies / 31)
        angles = timestep[:, None] * frequencies[None, :]
        time_features = torch.cat((angles.sin(), angles.cos()), dim=1)
        cond = self.time_embed(time_features)
        cond = cond.view(x.size(0), -1, 1, 1)
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        m = self.mid(x2)
        u1 = self.up1(m)
        u1 = u1 + x1
        u0 = self.up0(u1)
        u0 = u0 + x0
        out = self.out(u0)
        return out + 0.1 * cond.mean(dim=1, keepdim=True)


class Diffusion:
    def __init__(self, steps=500):
        self.steps = steps
        self.betas = torch.linspace(1e-4, 0.02, steps, device=DEVICE)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean, t, noise=None):
        if noise is None:
            noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[t].to(clean.device).view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise

    def sample(self, denoiser, count, steps=None):
        if steps is None:
            steps = self.steps
        denoiser.eval()
        x = torch.randn(count, 1, 32, 32, device=DEVICE)
        schedule = torch.linspace(self.steps - 1, 0, steps, device=DEVICE).round().long()
        for index, timestep in enumerate(schedule):
            t = torch.full((count,), timestep.item(), device=DEVICE, dtype=torch.long)
            with torch.no_grad():
                pred = denoiser(x, t)
            alpha_bar_t = self.alpha_bars[timestep]
            previous_timestep = schedule[index + 1] if index + 1 < len(schedule) else None
            alpha_bar_previous = torch.tensor(1.0, device=DEVICE) if previous_timestep is None else self.alpha_bars[previous_timestep]
            predicted_clean = (x - torch.sqrt(1.0 - alpha_bar_t) * pred) / torch.sqrt(alpha_bar_t)
            x = torch.sqrt(alpha_bar_previous) * predicted_clean + torch.sqrt(1.0 - alpha_bar_previous) * pred
        return x.clamp(-1, 1)


def train_diffusion(denoiser):
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-4)
    history = []
    diffusion = Diffusion(steps=CONFIG["diffusion_steps"])
    started = __import__("time").perf_counter()
    denoiser.train()
    batches = iter(train_loader)
    for step in range(CONFIG["diffusion_train_steps"]):
        try:
            images, _ = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            images, _ = next(batches)
        batch = images.to(DEVICE)
        t = torch.randint(0, diffusion.steps, (batch.size(0),), device=DEVICE)
        noise = torch.randn_like(batch)
        noisy = diffusion.add_noise(batch, t, noise=noise)
        pred = denoiser(noisy, t)
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(1, CONFIG["diffusion_train_steps"] // 10) == 0:
            history.append({"step": step + 1, "mse": loss.item()})
    return denoiser, diffusion, history, __import__("time").perf_counter() - started


@torch.no_grad()
def sample_diffusion(diffusion, denoiser, count, steps=None):
    return diffusion.sample(denoiser, count, steps=steps)
