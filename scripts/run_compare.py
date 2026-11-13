"""From scratch versus linear probe versus LoRA on Flowers-102.

Examples:
    uv run python scripts/run_compare.py --smoke
    uv run python scripts/run_compare.py
    uv run python scripts/run_compare.py --methods lora
    uv run python scripts/run_compare.py --plot-only
"""

import argparse

import torch

from vitfinetune import constants as C
from vitfinetune.experiments.transfer import (
    MethodCatalog, TransferMethod, TransferModelBuilder, TransferRunner, TransferSettings,
)
from vitfinetune.plotting import TransferPlotter
from vitfinetune.training import ResultStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", choices=[m.value for m in TransferMethod])
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=C.DEFAULT_NUM_WORKERS)
    parser.add_argument("--smoke", action="store_true", help="1 epoch on 128 images, to a separate folder")
    parser.add_argument("--plot-only", action="store_true", help="redraw the chart from saved results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = TransferSettings(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.workers)
    output_dir = C.RUNS_DIR
    if args.smoke:
        settings, output_dir = settings.smoke(), output_dir / "smoke"
    if not args.plot_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        runner = TransferRunner(settings, TransferModelBuilder(), C.DATA_DIR, output_dir, device)
        runner.run(MethodCatalog.select(args.methods))
    results = ResultStore(output_dir).load_available([m.value for m in TransferMethod])
    print("wrote", TransferPlotter().plot(results, output_dir / "comparison.png"))


if __name__ == "__main__":
    main()
