from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class SKNOConfig:
    input_shape: Tuple[int, int]
    latent_dim: int = 64
    hidden_dim: int = 128
    l1_lambda: float = 1e-5
    koopman_weight: float = 1.0
    recon_weight: float = 1.0
    spectral_weight: float = 1e-6
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 10

    @property
    def input_dim(self) -> int:
        return self.input_shape[0] * self.input_shape[1]


class ConvEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], hidden_dim: int, latent_dim: int):
        super().__init__()
        h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((max(4, h // 8), max(4, w // 8))),
        )
        pooled_h = max(4, h // 8)
        pooled_w = max(4, w // 8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * pooled_h * pooled_w, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class ConvDecoder(nn.Module):
    def __init__(self, output_shape: Tuple[int, int], hidden_dim: int, latent_dim: int):
        super().__init__()
        h, w = output_shape
        self.h = h
        self.w = w
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, h * w),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        return x.view(z.shape[0], 1, self.h, self.w)


class SparseKoopmanLayer(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        K0 = torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)
        self.K = nn.Parameter(K0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.K.T

    @torch.no_grad()
    def soft_threshold(self, lam: float):
        self.K.data = torch.sign(self.K.data) * torch.clamp(torch.abs(self.K.data) - lam, min=0.0)


class SKNO(nn.Module):
    def __init__(self, cfg: SKNOConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEncoder(cfg.input_shape, cfg.hidden_dim, cfg.latent_dim)
        self.koopman = SparseKoopmanLayer(cfg.latent_dim)
        self.decoder = ConvDecoder(cfg.input_shape, cfg.hidden_dim, cfg.latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def evolve_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.koopman(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x_t: torch.Tensor):
        z_t = self.encode(x_t)
        z_next_pred = self.evolve_latent(z_t)
        x_next_pred = self.decode(z_next_pred)
        return x_next_pred, z_t, z_next_pred


def spectral_penalty(K: torch.Tensor) -> torch.Tensor:
    return torch.norm(K, p="fro")


def loss_fn(model: SKNO, x_t: torch.Tensor, x_tp1: torch.Tensor, cfg: SKNOConfig):
    x_next_pred, z_t, z_next_pred = model(x_t)
    z_tp1_true = model.encode(x_tp1)

    pred_loss = F.mse_loss(x_next_pred, x_tp1)
    koopman_loss = F.mse_loss(z_next_pred, z_tp1_true)
    l1_loss = torch.sum(torch.abs(model.koopman.K))
    spec_loss = spectral_penalty(model.koopman.K)

    total = (
        cfg.recon_weight * pred_loss
        + cfg.koopman_weight * koopman_loss
        + cfg.l1_lambda * l1_loss
        + cfg.spectral_weight * spec_loss
    )
    return total, {
        "total": float(total.detach().cpu()),
        "pred": float(pred_loss.detach().cpu()),
        "koopman": float(koopman_loss.detach().cpu()),
        "l1": float(l1_loss.detach().cpu()),
        "spec": float(spec_loss.detach().cpu()),
    }


def train_step(model: SKNO, optimizer: torch.optim.Optimizer, x_t: torch.Tensor, x_tp1: torch.Tensor, cfg: SKNOConfig) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, metrics = loss_fn(model, x_t, x_tp1, cfg)
    total_loss.backward()
    optimizer.step()
    model.koopman.soft_threshold(cfg.lr * cfg.l1_lambda)
    return metrics


@torch.no_grad()
def evaluate_batch(model: SKNO, x_t: torch.Tensor, x_tp1: torch.Tensor, cfg: SKNOConfig) -> Dict[str, float]:
    model.eval()
    total_loss, metrics = loss_fn(model, x_t, x_tp1, cfg)
    metrics["total"] = float(total_loss.detach().cpu())
    return metrics


@torch.no_grad()
def rollout(model: SKNO, x0: torch.Tensor, steps: int) -> torch.Tensor:
    model.eval()
    xs = [x0]
    x = x0
    for _ in range(steps):
        x, _, _ = model(x)
        xs.append(x)
    return torch.cat(xs, dim=0)
