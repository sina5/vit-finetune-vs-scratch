"""Central constants for this repository.

Scripts and classes read their defaults from here so that one value has one home.
"""

from pathlib import Path

# --- Paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
RESULT_SUFFIX = ".json"

# --- General ------------------------------------------------------------------
DEFAULT_SEED = 0
DEFAULT_NUM_WORKERS = 6
BYTES_PER_MIB = 1024 * 1024

# --- ImageNet-style inputs ----------------------------------------------------
IMAGENET_IMAGE_SIZE = 224
IMAGENET_RESIZE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Fine-tune versus scratch -------------------------------------------------
PRETRAINED_MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
SCRATCH_MODEL = "vit_base_patch16_224"
NUM_CLASSES = 102  # Oxford Flowers-102
EPOCHS = 30
BATCH_SIZE = 32
EVAL_EVERY = 5
WARMUP_EPOCHS = 3
WEIGHT_DECAY = 0.05
SGD_MOMENTUM = 0.9
GRAD_CLIP_NORM = 1.0
SCRATCH_LR = 3e-4
PROBE_LR = 1e-3
LORA_LR = 1e-3
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.1
LORA_TARGETS = ("qkv",)
HEAD_NAME = "head"
SMOKE_EPOCHS = 1
SMOKE_SUBSET = 128

# --- Split overlap check ------------------------------------------------------
OVERLAP_DIR = RUNS_DIR / "overlap"
FLOWERS_DIR = DATA_DIR / "flowers-102"
SPLIT_KEYS = {"train": "trnid", "val": "valid", "test": "tstid"}  # keys in setid.mat
DHASH_SIZE = 8  # 8x8 = 64-bit difference hash
DHASH_NEAR = 6  # Hamming distance (of 64 bits) at or below which a pair is a near-duplicate candidate
COSINE_NEAR = 0.95  # embedding cosine similarity at or above which a pair is a candidate
REVIEW_PAIRS = 12  # pairs shown in each review grid
EMBED_BATCH = 64

# --- Plotting -----------------------------------------------------------------
FIG_SIZE = (16, 9)
FIG_DPI = 150
PALETTE = (
    "#E63946", "#1D77B4", "#2A9D8F", "#F4A261", "#8E44AD",
    "#264653", "#E9C46A", "#A0522D", "#06B6D4", "#65A30D",
)
