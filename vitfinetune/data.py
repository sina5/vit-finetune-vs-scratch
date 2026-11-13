"""Dataset and input pipeline for Oxford Flowers-102."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2

from vitfinetune import constants as C


class Flowers102Data:
    """Oxford Flowers-102 at 224x224 with ImageNet normalization.

    The official split has 10 training images per class (1,020 in total), which is
    the "small dataset" the comparison needs. Evaluation uses the 6,149 test images.
    """

    def __init__(self, root: Path, batch_size: int, num_workers: int, subset: int | None = None) -> None:
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset = subset

    def _train_transform(self) -> v2.Transform:
        return v2.Compose([
            v2.PILToTensor(),
            v2.RandomResizedCrop(C.IMAGENET_IMAGE_SIZE, antialias=True),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
        ])

    def _test_transform(self) -> v2.Transform:
        return v2.Compose([
            v2.PILToTensor(),
            v2.Resize(C.IMAGENET_RESIZE_SIZE, antialias=True),
            v2.CenterCrop(C.IMAGENET_IMAGE_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
        ])

    def _dataset(self, split: str, transform: v2.Transform) -> Dataset:
        dataset = datasets.Flowers102(str(self.root), split=split, transform=transform, download=True)
        if self.subset is not None:
            return Subset(dataset, range(min(self.subset, len(dataset))))
        return dataset

    def train_loader(self) -> DataLoader:
        """Shuffled, augmented training batches (downloads the data on first use)."""
        return self._loader(self._dataset("train", self._train_transform()), shuffle=True)

    def test_loader(self) -> DataLoader:
        """Test batches in a fixed order."""
        return self._loader(self._dataset("test", self._test_transform()), shuffle=False)

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
