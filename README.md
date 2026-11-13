# vit-finetune-vs-scratch

Fine-tune versus train from scratch.

One small dataset, one architecture (ViT-B/16, 86 M parameters), three ways to train it:

| Method | Starting weights | Trainable parameters |
|--------|------------------|----------------------|
| From scratch | random | all |
| Linear probe | pretrained | classifier head only |
| LoRA | pretrained | rank-8 LoRA on every `qkv` projection, plus the head |

For each method, the project reports test accuracy, training time, peak VRAM and the number of trainable parameters.

## Dataset

[Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/): 102 flower classes. The official training split has 10 images per class (1,020 images). The test split has 6,149 images. Torchvision downloads it to `data/` on first use.

## Settings

All three methods use the same data, augmentation (random resized crop, flip), 30 epochs, batch size 32, AdamW with 3 warmup epochs and cosine decay, and bfloat16 mixed precision. Only the learning rate changes: 3e-4 from scratch, 1e-3 for the probe and LoRA.

- Pretrained weights: `vit_base_patch16_224.augreg2_in21k_ft_in1k` from timm. They download from the Hugging Face Hub on first use (about 350 MB).
- LoRA is implemented in `vitfinetune/lora.py` (about 40 lines), not imported from a library, so the method stays visible.

## Setup

You need [uv](https://docs.astral.sh/uv/) and an NVIDIA driver that supports CUDA 12.8.

```bash
uv sync --group dev
uv run pytest
```

`uv sync` installs PyTorch with CUDA 12.8 from the PyTorch package index. The scripts download the dataset and the pretrained weights to `data/` on first use. They write results, charts and checkpoints to `runs/`. The repository includes the results and charts of the full runs in `runs/`. Git ignores `data/`, the model weights and the `smoke/` folders.

## Run

```bash
# Check the pipeline (1 epoch on 128 images)
uv run python scripts/run_compare.py --smoke

# Full comparison
uv run python scripts/run_compare.py

# One method; redraw the chart from saved results
uv run python scripts/run_compare.py --methods lora
uv run python scripts/run_compare.py --plot-only
```

## Output

In `runs/`:

- `<method>.json`: per-epoch metrics, wall time, peak VRAM, trainable and total parameters.
- `comparison.png`: accuracy, training time, peak VRAM and trainable parameters, side by side.

Training time includes the test evaluations (every 5 epochs). Peak VRAM is `torch.cuda.max_memory_allocated`, so it includes the model weights.

## Overlap check

`check_overlap.py` looks for val and test images that also appear in the training split. It uses three detectors: identical file bytes (SHA-256), a 64-bit perceptual hash (dHash, mirror images included), and cosine similarity of the pretrained ViT-B/16 embeddings. It runs on the CPU in about 15 minutes and writes `runs/overlap/` (a JSON report and review grids of the closest pairs).

```bash
uv run python scripts/check_overlap.py
```

Findings on the official split:

- **Exact copies**: 1 test image is byte-identical to a training image (test 7309 = train 7307). Across all 8,189 images, 4 pairs of files are byte-identical.
- **Perceptual hash**: 7 test images are within 6 bits of a training image. Apart from the exact copy, a review showed no copies: they are different photos (4 of them from different classes).
- **Embeddings**: 279 of the 6,149 test images (4.5%) have cosine similarity of 0.95 or more to a training image, and 278 of these are the same class. Among the highest pairs, many are near-duplicates: the same flower in the same scene, photographed a moment apart with a small change of framing. The others are only the same species. In val, 53 of 1,020 images are at or above 0.95.
- **Pretraining data**: this check cannot tell whether test images are in ImageNet-21k, which the pretrained weights saw. That data set is not available here.

Effect on the results: the near-duplicates are too few to change the conclusion. If all 279 flagged test images are removed and each one was classified correctly (the worst case), LoRA drops from 99.2% to at least 99.1%, and the linear probe from 98.4% to at least 98.3%. From scratch, the accuracy on the other 5,870 images is between 36.1% and 40.9%, depending on how many of the 279 the model got right. The exact numbers need the trained models, which this project does not save.

## Results

One run, final epoch, official test split (6,149 images):

| Method | Test accuracy | Training time | Peak VRAM | Trainable parameters |
|--------|---------------|---------------|-----------|----------------------|
| From scratch | 39.0% | 2.7 min | 3,578 MiB | 85.9 M |
| Linear probe | 98.4% | 1.6 min | 513 MiB | 78 K |
| LoRA r=8 | 99.2% | 2.3 min | 2,112 MiB | 373 K |

Hyperparameters were not tuned on the test split. The test split was evaluated every 5 epochs for the log only, and the table reports the final epoch, not the best one.

## Layout

```
vitfinetune/           package
  constants.py         every default value (epochs, learning rates, paths)
  data.py              Flowers-102 pipeline
  training.py          training loop, LR schedule, result files
  lora.py              LoRA for nn.Linear
  plotting.py          charts
  experiments/         the three training methods, the overlap check
scripts/               entry scripts
tests/                 pytest suite (CPU only, no downloads)
```

## License

The code is licensed under the [Apache License 2.0](LICENSE).
