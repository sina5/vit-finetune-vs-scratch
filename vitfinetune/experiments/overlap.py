"""Side check: do Flowers-102 test (or val) images also appear in the training split?

Three detectors, from strict to loose:

1. Exact: identical file bytes (SHA-256).
2. Perceptual: a 64-bit difference hash (dHash). It survives resizing and recompression.
   The mirror image is hashed too, so flipped copies are found.
3. Semantic: cosine similarity of pretrained ViT embeddings. This finds crops and edits that
   the hashes miss, but it also flags different photos of the same flower, so a person must
   review the candidate pairs (the script draws them).
"""

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageOps
from scipy.io import loadmat
from torch import nn

from vitfinetune import constants as C
from vitfinetune.errors import ConfigError
from vitfinetune.plotting import apply_style, save_png

IMAGE_PATTERN = "image_{:05d}.jpg"
LABELS_FILE = "imagelabels.mat"
SPLITS_FILE = "setid.mat"
IMAGE_DIR = "jpg"
HASH_DTYPE = np.uint64
PAIR_COLUMNS = 2
TITLE_SIZE = 9


@dataclass(frozen=True)
class FlowerImage:
    """One image of the official Flowers-102 release."""

    image_id: int  # 1-based id from the file name
    path: Path
    label: int  # 0-based class
    split: str


class FlowersIndex:
    """Reads the official split files (setid.mat, imagelabels.mat) without torchvision internals."""

    def __init__(self, root: Path = C.FLOWERS_DIR) -> None:
        if not (root / SPLITS_FILE).exists():
            raise ConfigError(f"{root / SPLITS_FILE} is missing; run scripts/run_compare.py once to download the data")
        self.root = root

    def split(self, name: str) -> list[FlowerImage]:
        """All images of one split ("train", "val" or "test"), in id order."""
        if name not in C.SPLIT_KEYS:
            raise ConfigError(f"unknown split {name!r}; use one of {sorted(C.SPLIT_KEYS)}")
        ids = sorted(int(i) for i in loadmat(self.root / SPLITS_FILE)[C.SPLIT_KEYS[name]].ravel())
        labels = loadmat(self.root / LABELS_FILE)["labels"].ravel()
        return [FlowerImage(i, self.root / IMAGE_DIR / IMAGE_PATTERN.format(i), int(labels[i - 1]) - 1, name)
                for i in ids]


class FileHasher:
    """SHA-256 of the raw file bytes: finds byte-identical copies."""

    def digest(self, path: Path) -> str:
        """Hex digest of one file."""
        return hashlib.sha256(path.read_bytes()).hexdigest()


class DifferenceHash:
    """dHash: compare neighboring pixels of a small grayscale thumbnail, one bit per comparison."""

    def __init__(self, hash_size: int = C.DHASH_SIZE) -> None:
        self.hash_size = hash_size

    def of_image(self, image: Image.Image) -> int:
        """The hash of one PIL image as an int with hash_size**2 bits."""
        thumb = image.convert("L").resize((self.hash_size + 1, self.hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(thumb, dtype=np.int16)
        bits = (pixels[:, 1:] > pixels[:, :-1]).ravel()
        return int(sum(1 << i for i, bit in enumerate(bits) if bit))

    def of_file(self, path: Path) -> tuple[int, int]:
        """Hashes of the image and of its mirror image."""
        with Image.open(path) as image:
            return self.of_image(image), self.of_image(ImageOps.mirror(image))


class HammingMatcher:
    """For each query hash, the closest reference hash by Hamming distance."""

    def nearest(self, queries: np.ndarray, references: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (distance, reference index) per query. Inputs are uint64 arrays."""
        if references.size == 0:
            raise ConfigError("the reference set is empty")
        distances = np.bitwise_count(queries[:, None] ^ references[None, :])
        best = distances.argmin(axis=1)
        return distances[np.arange(len(queries)), best].astype(int), best


class EmbeddingExtractor:
    """L2-normalized pooled features of a pretrained model, on the test-time view of each image."""

    def __init__(self, model: nn.Module, transform, device: torch.device, batch_size: int = C.EMBED_BATCH) -> None:
        self.model = model.eval().to(device)
        self.transform = transform
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def embed(self, paths: list[Path]) -> torch.Tensor:
        """Return an (N, D) float tensor on the CPU."""
        chunks = []
        for start in range(0, len(paths), self.batch_size):
            batch = torch.stack([self._load(p) for p in paths[start:start + self.batch_size]]).to(self.device)
            chunks.append(nn.functional.normalize(self.model(batch).float(), dim=1).cpu())
        return torch.cat(chunks)

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


class CosineMatcher:
    """For each query embedding, the most similar reference embedding."""

    def nearest(self, queries: torch.Tensor, references: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return (similarity, reference index) per query. Inputs must be L2-normalized."""
        similarity, best = (queries @ references.T).max(dim=1)
        return similarity.numpy(), best.numpy()


@dataclass(frozen=True)
class MatchPair:
    """A query image and its closest training image under one detector."""

    query_id: int
    query_label: int
    train_id: int
    train_label: int
    score: float  # Hamming distance for dHash, cosine similarity for embeddings


@dataclass
class SplitReport:
    """Overlap of one evaluation split with the training split."""

    split: str
    images: int
    exact_duplicates: list[MatchPair] = field(default_factory=list)
    dhash_candidates: list[MatchPair] = field(default_factory=list)
    cosine_candidates: list[MatchPair] = field(default_factory=list)
    cosine_top: list[MatchPair] = field(default_factory=list)  # the highest pairs, for review

    def to_dict(self) -> dict:
        """JSON-ready form."""
        return asdict(self)


class SplitOverlapChecker:
    """Compares one evaluation split against the training split with all three detectors."""

    def __init__(self, extractor: EmbeddingExtractor, review_pairs: int = C.REVIEW_PAIRS,
                 dhash_near: int = C.DHASH_NEAR, cosine_near: float = C.COSINE_NEAR) -> None:
        self.extractor = extractor
        self.review_pairs = review_pairs
        self.dhash_near = dhash_near
        self.cosine_near = cosine_near
        self.files = FileHasher()
        self.dhash = DifferenceHash()

    def check(self, train: list[FlowerImage], queries: list[FlowerImage],
              train_embeddings: torch.Tensor) -> SplitReport:
        """Run every detector and collect the pairs worth a look."""
        report = SplitReport(queries[0].split if queries else "", len(queries))
        report.exact_duplicates = self._exact(train, queries)
        report.dhash_candidates = self._perceptual(train, queries)
        similarity, best = CosineMatcher().nearest(self.extractor.embed([q.path for q in queries]), train_embeddings)
        pairs = [self._pair(queries[i], train[best[i]], float(similarity[i])) for i in np.argsort(-similarity)]
        report.cosine_candidates = [p for p in pairs if p.score >= self.cosine_near]
        report.cosine_top = pairs[:self.review_pairs]
        return report

    def _exact(self, train: list[FlowerImage], queries: list[FlowerImage]) -> list[MatchPair]:
        by_digest = {self.files.digest(t.path): t for t in train}
        return [self._pair(q, by_digest[d], 0.0) for q in queries if (d := self.files.digest(q.path)) in by_digest]

    def _perceptual(self, train: list[FlowerImage], queries: list[FlowerImage]) -> list[MatchPair]:
        train_hashes = np.array([self.dhash.of_file(t.path)[0] for t in train], dtype=HASH_DTYPE)
        query_hashes = np.array([self.dhash.of_file(q.path) for q in queries], dtype=HASH_DTYPE)
        matcher = HammingMatcher()
        plain, plain_idx = matcher.nearest(query_hashes[:, 0], train_hashes)
        mirror, mirror_idx = matcher.nearest(query_hashes[:, 1], train_hashes)
        distance = np.minimum(plain, mirror)
        index = np.where(plain <= mirror, plain_idx, mirror_idx)
        order = np.argsort(distance, kind="stable")
        return [self._pair(queries[i], train[index[i]], float(distance[i])) for i in order
                if distance[i] <= self.dhash_near]

    @staticmethod
    def _pair(query: FlowerImage, match: FlowerImage, score: float) -> MatchPair:
        return MatchPair(query.image_id, query.label, match.image_id, match.label, score)


class PairGridPlotter:
    """Draws query/train pairs side by side so a person can judge them."""

    def plot(self, pairs: list[MatchPair], root: Path, title: str, score_name: str, path: Path) -> Path:
        """One cell per pair: query on the left, closest training image on the right."""
        apply_style()
        fig = Figure(figsize=C.FIG_SIZE)
        rows, cols = self._grid(len(pairs))
        axes = fig.subplots(rows, cols * PAIR_COLUMNS, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for n, pair in enumerate(pairs):
            row, col = divmod(n, cols)
            self._show(axes[row, col * PAIR_COLUMNS], root, pair.query_id, f"query {pair.query_id} (c{pair.query_label})")
            self._show(axes[row, col * PAIR_COLUMNS + 1], root, pair.train_id,
                       f"train {pair.train_id} (c{pair.train_label}) {score_name}={pair.score:.3g}")
        fig.suptitle(title, fontsize=TITLE_SIZE + 7)
        fig.tight_layout()
        return save_png(fig, path)

    @staticmethod
    def _grid(count: int) -> tuple[int, int]:
        cols = 3
        return max(1, -(-count // cols)), cols

    @staticmethod
    def _show(ax, root: Path, image_id: int, label: str) -> None:
        with Image.open(root / IMAGE_DIR / IMAGE_PATTERN.format(image_id)) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(label, fontsize=TITLE_SIZE)
