"""
Examples:
    python main.py all
    python main.py polynomial
    python main.py purity --epochs 1000
    python main.py entanglement --device cuda
    python main.py bloch --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from config import CLASSIFICATION_NUM_EPOCHS, DEFAULT_SEED, LEARNING_RATE, POLY_NUM_EPOCHS
from experiments import (
    run_bloch_sphere_classification,
    run_entanglement_entropy_classification,
    run_polynomial_fitting,
    run_purity_classification,
)


def parse_seed(value: str) -> Optional[int]:
    if value.lower() in {"none", "null", "off"}:
        return None
    return int(value)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the re-uploading paper.")
    parser.add_argument(
        "experiment",
        choices=["all", "polynomial", "purity", "entanglement", "bloch"],
        nargs="?",
        default="all",
        help="Which experiment group to run.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for generated plots.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--seed", default=str(DEFAULT_SEED), help="Integer seed, or 'none' to disable seeding.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for the selected experiment. Defaults: 10000 for polynomial, 1000 for classification.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Run training without saving plots.")
    parser.add_argument("--print-every", type=int, default=None, help="Progress print interval. Use 0 to silence.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    seed = parse_seed(args.seed)
    make_plots = not args.no_plots
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Output directory: {args.output_dir.resolve()}")

    if args.experiment in {"all", "polynomial"}:
        epochs = POLY_NUM_EPOCHS if args.epochs is None else args.epochs
        print_every = 100 if args.print_every is None else args.print_every
        summary = run_polynomial_fitting(
            output_dir=args.output_dir,
            device=device,
            epochs=epochs,
            lr=args.lr,
            seed=seed,
            make_plots=make_plots,
            print_every=print_every,
        )
        print("Polynomial summary:", summary)

    if args.experiment in {"all", "purity"}:
        epochs = CLASSIFICATION_NUM_EPOCHS if args.epochs is None else args.epochs
        print_every = 50 if args.print_every is None else args.print_every
        summary = run_purity_classification(
            output_dir=args.output_dir,
            device=device,
            epochs=epochs,
            lr=args.lr,
            seed=seed,
            make_plots=make_plots,
            print_every=print_every,
        )
        print("Purity classification best test accuracies:", summary)

    if args.experiment in {"all", "entanglement"}:
        epochs = CLASSIFICATION_NUM_EPOCHS if args.epochs is None else args.epochs
        print_every = 50 if args.print_every is None else args.print_every
        summary = run_entanglement_entropy_classification(
            output_dir=args.output_dir,
            device=device,
            epochs=epochs,
            lr=args.lr,
            seed=seed,
            make_plots=make_plots,
            print_every=print_every,
        )
        print("Entanglement entropy classification best test accuracies:", summary)

    if args.experiment in {"all", "bloch"}:
        epochs = CLASSIFICATION_NUM_EPOCHS if args.epochs is None else args.epochs
        print_every = 50 if args.print_every is None else args.print_every
        summary = run_bloch_sphere_classification(
            output_dir=args.output_dir,
            device=device,
            epochs=epochs,
            lr=args.lr,
            seed=seed,
            make_plots=make_plots,
            print_every=print_every,
        )
        print("Bloch sphere classification best test accuracies:", summary)


if __name__ == "__main__":
    main()
