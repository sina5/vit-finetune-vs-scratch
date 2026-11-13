"""Side check: look for test and val images that also appear in the Flowers-102 training split.

Examples:
    uv run python scripts/check_overlap.py
    uv run python scripts/check_overlap.py --device cuda
"""

import argparse
import json
from collections import defaultdict

import timm
import torch

from vitfinetune import constants as C
from vitfinetune.experiments.overlap import (
    EmbeddingExtractor, FileHasher, FlowersIndex, PairGridPlotter, SplitOverlapChecker,
)

EVAL_SPLITS = ("val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu (default, leaves the GPU free) or cuda")
    parser.add_argument("--threads", type=int, default=C.DEFAULT_NUM_WORKERS, help="CPU threads for torch")
    return parser.parse_args()


def build_extractor(device: torch.device) -> EmbeddingExtractor:
    """The same pretrained ViT-B/16 that run_compare.py fine-tunes, without its classifier."""
    model = timm.create_model(C.PRETRAINED_MODEL, pretrained=True, num_classes=0)
    config = timm.data.resolve_data_config({}, model=model)
    return EmbeddingExtractor(model, timm.data.create_transform(**config), device)


def duplicate_files_anywhere(index: FlowersIndex) -> list[list[int]]:
    """Groups of byte-identical files across all 8,189 images, any split."""
    groups: dict[str, list[int]] = defaultdict(list)
    hasher = FileHasher()
    for split in C.SPLIT_KEYS:
        for image in index.split(split):
            groups[hasher.digest(image.path)].append(image.image_id)
    return [ids for ids in groups.values() if len(ids) > 1]


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    index, out = FlowersIndex(), C.OVERLAP_DIR
    extractor = build_extractor(torch.device(args.device))
    train = index.split("train")
    train_embeddings = extractor.embed([t.path for t in train])
    checker, plotter = SplitOverlapChecker(extractor), PairGridPlotter()
    summary = {"byte_identical_groups_any_split": duplicate_files_anywhere(index), "splits": {}}
    for split in EVAL_SPLITS:
        report = checker.check(train, index.split(split), train_embeddings)
        summary["splits"][split] = report.to_dict()
        print(f"{split}: {report.images} images, exact={len(report.exact_duplicates)}, "
              f"dHash<={C.DHASH_NEAR}: {len(report.dhash_candidates)}, "
              f"cosine>={C.COSINE_NEAR}: {len(report.cosine_candidates)}, "
              f"top cosine={report.cosine_top[0].score:.4f}")
        plotter.plot(report.cosine_top, index.root, f"{split}: closest training image by ViT embedding",
                     "cos", out / f"{split}_top_cosine.png")
        if report.dhash_candidates:
            plotter.plot(report.dhash_candidates[:C.REVIEW_PAIRS], index.root,
                         f"{split}: closest training image by dHash", "hamming", out / f"{split}_dhash.png")
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlap.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("byte-identical groups (any split):", summary["byte_identical_groups_any_split"])
    print("wrote", out)


if __name__ == "__main__":
    main()
