from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_loss_curves(history: Dict[str, Iterable[float]], out_path: Path):
    plt.figure(figsize=(6, 4))
    for k in ["train_total", "val_total", "train_pred", "val_pred", "train_koopman", "val_koopman"]:
        if k in history and len(history[k]) > 0:
            plt.plot(history[k], label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_koopman_matrix(K, out_path: Path):
    K = _to_np(K)
    plt.figure(figsize=(5, 4))
    plt.imshow(K, aspect="auto")
    plt.colorbar()
    plt.title("Learned Sparse Koopman Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_qualitative_prediction(x_true, x_pred, out_path: Path):
    x_true = _to_np(x_true).squeeze()
    x_pred = _to_np(x_pred).squeeze()
    err = np.abs(x_true - x_pred)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    ims = []
    ims.append(axes[0].imshow(x_true))
    axes[0].set_title("Ground Truth")
    ims.append(axes[1].imshow(x_pred))
    axes[1].set_title("Prediction")
    ims.append(axes[2].imshow(err))
    axes[2].set_title("Abs. Error")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.75)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_rollout(truth_seq, pred_seq, out_path: Path, max_steps: int = 4):
    truth_seq = _to_np(truth_seq)
    pred_seq = _to_np(pred_seq)
    T = min(max_steps, len(pred_seq), len(truth_seq))
    fig, axes = plt.subplots(3, T, figsize=(3 * T, 7))
    for t in range(T):
        gt = truth_seq[t].squeeze()
        pr = pred_seq[t].squeeze()
        er = np.abs(gt - pr)
        axes[0, t].imshow(gt)
        axes[0, t].set_title(f"Truth t+{t}")
        axes[1, t].imshow(pr)
        axes[1, t].set_title(f"Pred t+{t}")
        axes[2, t].imshow(er)
        axes[2, t].set_title(f"Error t+{t}")
        for r in range(3):
            axes[r, t].set_xticks([])
            axes[r, t].set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_neurips_panel(loss_png: Path, koopman_png: Path, qualitative_png: Path, rollout_png: Path, out_path: Path):
    from PIL import Image

    imgs = [Image.open(p).convert("RGB") for p in [loss_png, koopman_png, qualitative_png, rollout_png]]
    widths, heights = zip(*(im.size for im in imgs))
    cell_w = max(widths)
    cell_h = max(heights)
    canvas = Image.new("RGB", (2 * cell_w, 2 * cell_h), color=(255, 255, 255))
    positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]
    for im, pos in zip(imgs, positions):
        canvas.paste(im.resize((cell_w, cell_h)), pos)
    canvas.save(out_path)
