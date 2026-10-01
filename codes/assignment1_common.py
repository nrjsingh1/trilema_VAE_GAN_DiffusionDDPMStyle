import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

CONFIG = {
    "full_run": False,
    "seed": 559,
    "batch_size": 128,
    "eval_epochs": 1,
    "vae_epochs": 1,
    "gan_epochs": 1,
    "diffusion_steps": 500,
    "diffusion_train_steps": 300,
    "metric_count": 500,
    "sample_count": 100,
    "num_workers": 0,
}


def configure_run(full_run=False):
    CONFIG["full_run"] = full_run
    if full_run:
        CONFIG.update({
            "eval_epochs": 5,
            "vae_epochs": 25,
            "gan_epochs": 25,
            "diffusion_train_steps": 3000,
            "metric_count": 2000,
        })
    else:
        CONFIG.update({
            "eval_epochs": 1,
            "vae_epochs": 1,
            "gan_epochs": 1,
            "diffusion_train_steps": 300,
            "metric_count": 500,
        })


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path.cwd()
OUT = ROOT / "assignment1_outputs"
OUT.mkdir(exist_ok=True)


def set_seed(seed: int = 559):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])


def make_loaders(batch_size=None, num_workers=None):
    bs = batch_size or CONFIG["batch_size"]
    workers = num_workers if num_workers is not None else CONFIG["num_workers"]
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.FashionMNIST(ROOT / "data", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(ROOT / "data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True, num_workers=workers)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=workers)
    return train_loader, test_loader


train_loader, test_loader = make_loaders()


class EvaluatorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x, return_features=False):
        feat = self.features(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat.view(feat.size(0), -1)
        return logits


def show_grid(images, title, filename, columns=10):
    images = ((images.detach().cpu().clamp(-1, 1) + 1) / 2).numpy()
    rows = max(1, math.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns, rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx < len(images):
            ax.imshow(images[idx, 0], cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def sqrt_psd(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    values = np.clip(values, 0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def coverage_metrics(real_features, real_labels, generated_features, generated_labels):
    real_mean = real_features.mean(0)
    gen_mean = generated_features.mean(0)
    real_cov = np.cov(real_features, rowvar=False) + 1e-6 * np.eye(real_features.shape[1])
    gen_cov = np.cov(generated_features, rowvar=False) + 1e-6 * np.eye(generated_features.shape[1])
    term = real_cov + gen_cov - 2 * sqrt_psd(real_cov @ gen_cov)
    cfid = np.sum((real_mean - gen_mean) ** 2) + np.trace(term)

    real_hist = np.bincount(real_labels, minlength=10).astype(np.float64)
    real_hist /= real_hist.sum()
    gen_hist = np.bincount(generated_labels, minlength=10).astype(np.float64)
    gen_hist /= gen_hist.sum()
    entropy = -np.sum(gen_hist * np.log(gen_hist + 1e-8))
    kl = np.sum(gen_hist * np.log((gen_hist + 1e-8) / (real_hist + 1e-8)))
    return {
        "cfid": float(cfid),
        "entropy": float(entropy),
        "kl_divergence": float(kl),
        "generated_histogram": gen_hist.tolist(),
    }


@torch.no_grad()
def evaluator_outputs(images, evaluator):
    logits, features = evaluator(images.to(DEVICE), return_features=True)
    return logits.argmax(1).cpu(), features.cpu()


def evaluator_accuracy(model):
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        logits = model(images.to(DEVICE))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += len(labels)
    return correct / total


def reference_features(count, evaluator):
    all_features, all_labels = [], []
    seen = 0
    for images, labels in test_loader:
        batch = min(len(images), count - seen)
        if batch <= 0:
            break
        _, features = evaluator_outputs(images[:batch], evaluator)
        all_features.append(features)
        all_labels.append(labels[:batch])
        seen += batch
    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def predicted_histogram(images, evaluator):
    labels = []
    for start in range(0, len(images), CONFIG["batch_size"]):
        batch_labels, _ = evaluator_outputs(images[start:start + CONFIG["batch_size"]], evaluator)
        labels.append(batch_labels)
    hist = torch.bincount(torch.cat(labels), minlength=10).float()
    return hist / hist.sum()


def train_evaluator(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    history = []
    model.train()
    for epoch in range(CONFIG["eval_epochs"]):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"evaluator {epoch + 1}", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        history.append({"epoch": epoch + 1, "loss": running_loss / total, "acc": correct / total})
    return model, history


def evaluate_model(name, sample_fn, evaluator, training_seconds=None, parameters=None):
    count = CONFIG["metric_count"]
    started = time.perf_counter()
    generated = sample_fn(count)
    elapsed = time.perf_counter() - started
    labels, features = evaluator_outputs(generated, evaluator)
    real_features, real_labels = reference_features(count, evaluator)
    metrics = coverage_metrics(real_features, real_labels, features.numpy(), labels.numpy())
    metrics.update({
        "model": name,
        "images_per_second": count / elapsed,
        "time_for_5000_images": 5000 * elapsed / count,
        "parameters": parameters,
        "training_seconds": training_seconds,
    })
    return metrics, generated


# must be imported in notebooks after time module exists
import time
