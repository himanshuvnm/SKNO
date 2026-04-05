from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .datasets import WindowConfig, build_dataset
from .figures import (
    save_koopman_matrix,
    save_loss_curves,
    save_neurips_panel,
    save_qualitative_prediction,
    save_rollout,
)
from .model import SKNO, SKNOConfig, evaluate_batch, get_device, rollout, train_step


class ExperimentRunner:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _aggregate(metrics_list):
        keys = metrics_list[0].keys()
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

    def run(
        self,
        dataset_name: str,
        root: Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        firms_map_key: Optional[str] = None,
        navier_match: str = "Navier",
        latent_dim: int = 64,
        hidden_dim: int = 128,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 1e-3,
        l1_lambda: float = 1e-5,
        koopman_weight: float = 1.0,
        recon_weight: float = 1.0,
        spectral_weight: float = 1e-6,
        train_fraction: float = 0.8,
        seed: int = 0,
    ) -> Dict[str, float]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = get_device()

        ds = build_dataset(
            dataset_name=dataset_name,
            root=Path(root),
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
            firms_map_key=firms_map_key,
            navier_match=navier_match,
            window=WindowConfig(input_steps=1, target_steps=1, stride=1),
        )
        sample_x, _ = ds[0]
        input_shape = tuple(sample_x.shape[-2:])

        cfg = SKNOConfig(
            input_shape=input_shape,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            l1_lambda=l1_lambda,
            koopman_weight=koopman_weight,
            recon_weight=recon_weight,
            spectral_weight=spectral_weight,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
        )

        n_train = max(1, int(len(ds) * train_fraction))
        n_val = max(1, len(ds) - n_train)
        if n_train + n_val > len(ds):
            n_train = len(ds) - 1
            n_val = 1
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = SKNO(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        history = {
            "train_total": [], "train_pred": [], "train_koopman": [],
            "val_total": [], "val_pred": [], "val_koopman": [],
        }

        best_val = float("inf")
        best_state = None

        for epoch in range(epochs):
            train_metrics = []
            for x_t, x_tp1 in train_loader:
                x_t = x_t.to(device)
                x_tp1 = x_tp1.to(device)
                train_metrics.append(train_step(model, optimizer, x_t, x_tp1, cfg))
            train_agg = self._aggregate(train_metrics)

            val_metrics = []
            for x_t, x_tp1 in val_loader:
                x_t = x_t.to(device)
                x_tp1 = x_tp1.to(device)
                val_metrics.append(evaluate_batch(model, x_t, x_tp1, cfg))
            val_agg = self._aggregate(val_metrics)

            history["train_total"].append(train_agg["total"])
            history["train_pred"].append(train_agg["pred"])
            history["train_koopman"].append(train_agg["koopman"])
            history["val_total"].append(val_agg["total"])
            history["val_pred"].append(val_agg["pred"])
            history["val_koopman"].append(val_agg["koopman"])

            if val_agg["total"] < best_val:
                best_val = val_agg["total"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            print(
                f"epoch={epoch+1:03d} train_total={train_agg['total']:.6f} "
                f"val_total={val_agg['total']:.6f} train_pred={train_agg['pred']:.6f} "
                f"val_pred={val_agg['pred']:.6f}"
            )

        if best_state is not None:
            model.load_state_dict(best_state)

        checkpoint_path = self.output_dir / "checkpoint.pt"
        torch.save({"model_state": model.state_dict(), "cfg": asdict(cfg)}, checkpoint_path)

        # Final evaluation + figures.
        x_t_vis, x_tp1_vis = val_ds[0]
        x_t_vis = x_t_vis.unsqueeze(0).to(device)
        x_tp1_vis = x_tp1_vis.unsqueeze(0).to(device)
        pred, _, _ = model(x_t_vis)

        truth_seq = [x_tp1_vis[0].detach().cpu()]
        pred_roll = rollout(model, x_t_vis, steps=min(3, len(val_ds) - 1)).detach().cpu()[1:]
        pred_seq = [pred_roll[i] for i in range(len(pred_roll))]
        # Ground-truth rollout proxy from validation set in sequence order.
        for i in range(1, min(4, len(val_ds))):
            _, y = val_ds[i]
            truth_seq.append(y)
        truth_seq = torch.stack(truth_seq[:len(pred_seq)], dim=0)
        pred_seq = torch.stack(pred_seq[:len(truth_seq)], dim=0)

        metrics = {
            "best_val_total": best_val,
            "final_train_total": history["train_total"][-1],
            "final_val_total": history["val_total"][-1],
            "final_val_pred": history["val_pred"][-1],
            "koopman_nnz_fraction": float((model.koopman.K.detach().abs() > 1e-6).float().mean().cpu()),
            "device": str(device),
            "dataset": dataset_name,
        }

        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        loss_png = self.output_dir / "loss_curves.png"
        koopman_png = self.output_dir / "koopman_matrix.png"
        qual_png = self.output_dir / "qualitative_prediction.png"
        rollout_png = self.output_dir / "rollout.png"
        panel_png = self.output_dir / "neurips_panel.png"

        save_loss_curves(history, loss_png)
        save_koopman_matrix(model.koopman.K, koopman_png)
        save_qualitative_prediction(x_tp1_vis[0], pred[0], qual_png)
        save_rollout(truth_seq, pred_seq, rollout_png)
        save_neurips_panel(loss_png, koopman_png, qual_png, rollout_png, panel_png)

        return metrics
