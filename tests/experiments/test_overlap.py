from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps
from scipy.io import savemat
from torch import nn

from vitfinetune.errors import ConfigError
from vitfinetune.experiments.overlap import (
    CosineMatcher, DifferenceHash, EmbeddingExtractor, FileHasher, FlowersIndex, HammingMatcher,
    PairGridPlotter, SplitOverlapChecker,
)

SIZE = 32


def _gradient_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8))


class _MeanColor(nn.Module):
    """Tiny stand-in embedder: per-channel means plus a fixed offset."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3)) + 1.0


def _to_tensor(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image.resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1)


@pytest.fixture
def flowers_root(tmp_path: Path) -> Path:
    """A fake release: 4 train, 2 test. Test 5 is a resized copy of train 1; test 6 is new."""
    (tmp_path / "jpg").mkdir()
    for i in range(1, 5):
        _gradient_image(i).save(tmp_path / "jpg" / f"image_{i:05d}.jpg", quality=95)
    _gradient_image(1).resize((SIZE * 2, SIZE * 2)).save(tmp_path / "jpg" / "image_00005.jpg", quality=95)
    _gradient_image(99).save(tmp_path / "jpg" / "image_00006.jpg", quality=95)
    savemat(tmp_path / "setid.mat", {"trnid": np.array([[1, 2, 3, 4]]), "valid": np.array([[6]]),
                                     "tstid": np.array([[5, 6]])})
    savemat(tmp_path / "imagelabels.mat", {"labels": np.array([[1, 2, 1, 2, 1, 3]])})
    return tmp_path


def test_index_reads_splits_and_zero_based_labels(flowers_root: Path) -> None:
    test = FlowersIndex(flowers_root).split("test")
    assert [(t.image_id, t.label, t.split) for t in test] == [(5, 0, "test"), (6, 2, "test")]


def test_index_rejects_unknown_split_and_missing_root(flowers_root: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        FlowersIndex(flowers_root).split("holdout")
    with pytest.raises(ConfigError):
        FlowersIndex(tmp_path / "nothing")


def test_dhash_is_stable_under_resizing_and_differs_for_other_images() -> None:
    dhash, image = DifferenceHash(), _gradient_image(3)
    same = dhash.of_image(image) ^ dhash.of_image(image.resize((SIZE * 3, SIZE * 3)))
    other = dhash.of_image(image) ^ dhash.of_image(_gradient_image(4))
    assert same.bit_count() <= 4 < other.bit_count()


def test_dhash_of_file_returns_the_mirror_hash(tmp_path: Path) -> None:
    path, image = tmp_path / "a.png", _gradient_image(5)
    image.save(path)
    plain, mirror = DifferenceHash().of_file(path)
    assert mirror == DifferenceHash().of_image(ImageOps.mirror(image)) and plain != mirror


def test_hamming_matcher_finds_the_closest_hash() -> None:
    distance, index = HammingMatcher().nearest(np.array([0b1011], dtype=np.uint64),
                                               np.array([0b0000, 0b1001, 0b1111], dtype=np.uint64))
    assert (distance.tolist(), index.tolist()) == ([1], [1])


def test_hamming_matcher_rejects_empty_reference() -> None:
    with pytest.raises(ConfigError):
        HammingMatcher().nearest(np.array([1], dtype=np.uint64), np.array([], dtype=np.uint64))


def test_cosine_matcher_finds_the_most_similar_row() -> None:
    queries = nn.functional.normalize(torch.tensor([[1.0, 0.1]]), dim=1)
    references = nn.functional.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), dim=1)
    similarity, index = CosineMatcher().nearest(queries, references)
    assert index.tolist() == [1] and similarity[0] > 0.99


def test_file_hasher_matches_identical_bytes_only(tmp_path: Path) -> None:
    for name, data in (("a", b"x"), ("b", b"x"), ("c", b"y")):
        (tmp_path / name).write_bytes(data)
    digests = [FileHasher().digest(tmp_path / n) for n in "abc"]
    assert digests[0] == digests[1] != digests[2]


def test_checker_flags_the_resized_copy_and_not_the_new_image(flowers_root: Path) -> None:
    index = FlowersIndex(flowers_root)
    train, test = index.split("train"), index.split("test")
    extractor = EmbeddingExtractor(_MeanColor(), _to_tensor, torch.device("cpu"), batch_size=2)
    report = SplitOverlapChecker(extractor, review_pairs=2).check(train, test, extractor.embed([t.path for t in train]))
    assert [(p.query_id, p.train_id) for p in report.dhash_candidates] == [(5, 1)]
    assert report.exact_duplicates == [] and report.images == 2 and len(report.cosine_top) == 2
    assert report.cosine_top[0].query_id == 5 and report.cosine_top[0].train_id == 1


def test_pair_grid_writes_a_png(flowers_root: Path, tmp_path: Path) -> None:
    index = FlowersIndex(flowers_root)
    extractor = EmbeddingExtractor(_MeanColor(), _to_tensor, torch.device("cpu"))
    train = index.split("train")
    report = SplitOverlapChecker(extractor).check(train, index.split("test"), extractor.embed([t.path for t in train]))
    path = PairGridPlotter().plot(report.cosine_top, flowers_root, "t", "cos", tmp_path / "grid.png")
    assert path.read_bytes()[:4] == b"\x89PNG"
