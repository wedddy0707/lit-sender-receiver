import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from lit_sender_receiver.base.dataset_base import DatasetBase
from lit_sender_receiver.base.variable_types import Batch, Sample


class OneHotDataset(DatasetBase):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def __getitem__(self, index: int) -> Sample:
        x = F.one_hot(torch.as_tensor(index), self.num_classes).float()
        return {"source": x, "target": x}

    def __len__(self) -> int:
        return self.num_classes


class OneHotDataModule(LightningDataModule):
    def __init__(self, num_classes: int, batch_size: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dataset = OneHotDataset(num_classes)

    def train_dataloader(self) -> DataLoader[Batch]:
        sampler = RandomSampler(
            self.dataset, replacement=True, num_samples=self.batch_size
        )
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler)

    def val_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(self.dataset, batch_size=self.batch_size)
