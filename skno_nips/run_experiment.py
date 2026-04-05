from __future__ import annotations

import argparse
from pathlib import Path

from skno.runner import ExperimentRunner


def parse_args():
    p = argparse.ArgumentParser(description="Run a compact SKNO experiment.")
    p.add_argument("--dataset", choices=["navier_stokes", "wildfire", "atmospheric"], required=True)
    p.add_argument("--root", type=Path, default=Path("./data"))
    p.add_argument("--output", type=Path, default=Path("./outputs/default_run"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l1-lambda", type=float, default=1e-5)
    p.add_argument("--koopman-weight", type=float, default=1.0)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--spectral-weight", type=float, default=1e-6)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--firms-map-key", type=str, default=None)
    p.add_argument("--navier-match", type=str, default="Navier")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    runner = ExperimentRunner(args.output)
    metrics = runner.run(
        dataset_name=args.dataset,
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        bbox=tuple(args.bbox) if args.bbox is not None else None,
        firms_map_key=args.firms_map_key,
        navier_match=args.navier_match,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_lambda=args.l1_lambda,
        koopman_weight=args.koopman_weight,
        recon_weight=args.recon_weight,
        spectral_weight=args.spectral_weight,
        seed=args.seed,
    )
    print("Done. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
