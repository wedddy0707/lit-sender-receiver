from abc import abstractmethod
from typing import Any

from lightning import LightningDataModule

from .dataset_base import DatasetBase


class DataModuleBase(LightningDataModule):
    @property
    @abstractmethod
    def train_dataset(self) -> DatasetBase:
        raise NotImplementedError()

    @property
    @abstractmethod
    def val_dataset(self) -> tuple[DatasetBase, ...]:
        raise NotImplementedError()

    def train_dataloader(self) -> Any:
        return super().train_dataloader()

    def val_dataloader(self) -> Any:
        return super().val_dataloader()
