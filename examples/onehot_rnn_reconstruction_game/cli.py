from lightning.pytorch.cli import LightningCLI

from .data import OneHotDataModule
from .model import OneHotRNNReconstructionGame


def cli_main() -> None:
    LightningCLI(OneHotRNNReconstructionGame, OneHotDataModule)


if __name__ == "__main__":
    print("This file is not supposed to be run as __main__.")
