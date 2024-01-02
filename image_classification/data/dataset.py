import logging
from pathlib import Path
from typing import Callable, Optional

from attr import define
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_logger = logging.getLogger(Path(__file__).stem)


@define
class Config:
    root_dir: Path  # The root directory which contains the train and test directories. Those directories contain the images.
    train_val_ratio: float
    train_transform: Optional[Callable]
    val_transform: Optional[Callable]

    tiny_dataset: bool = False  # If true, only use 5% of the data.


class CatsVsDogsDataset(Dataset):
    def __init__(self, file_list: list[Path], transform: Callable | None = None):
        self.file_list = file_list

        num_cats = sum(["cat" in str(f.stem) for f in file_list])
        num_dogs = sum(["dog" in str(f.stem) for f in file_list])
        if num_cats + num_dogs != len(file_list):
            raise ValueError("Not all files are cats or dogs, please check the file names")
        _logger.info(f"{self.__class__.__name__} with {num_cats} cats and {num_dogs} dogs")

        self.to_tensor = transforms.ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> tuple[Image.Image, int]:
        if idx < 0 or idx >= len(self.file_list):
            raise IndexError(f"Index {idx} out of range")

        image = Image.open(self.file_list[idx])
        label = 0 if "cat" in str(self.file_list[idx].stem) else 1
        if self.transform:
            image = self.transform(image)
        else:
            # If no transform is provided, convert to tensor
            image = self.to_tensor(image)
        return image, label


def create_train_val_datasets(config: Config) -> tuple[CatsVsDogsDataset, CatsVsDogsDataset]:
    train_dir = config.root_dir / "train"
    dog_files = sorted(list(train_dir.glob("dog*.jpg")))
    cat_files = sorted(list(train_dir.glob("cat*.jpg")))

    if config.tiny_dataset:
        tiny_dataset_ratio = 0.05
        dog_files = dog_files[: int(len(dog_files) * tiny_dataset_ratio)]
        cat_files = cat_files[: int(len(cat_files) * tiny_dataset_ratio)]

    train_files = (
        dog_files[: int(len(dog_files) * config.train_val_ratio)]
        + cat_files[: int(len(cat_files) * config.train_val_ratio)]
    )
    val_files = (
        dog_files[int(len(dog_files) * config.train_val_ratio) :]
        + cat_files[int(len(cat_files) * config.train_val_ratio) :]
    )

    return (
        CatsVsDogsDataset(train_files, config.train_transform),
        CatsVsDogsDataset(val_files, config.val_transform),
    )
