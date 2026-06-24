from typing import Literal

import pytest
import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from lightning import LightningDataModule, Trainer
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler

from lit_sender_receiver.agents.rnn_receiver import RNNReceiver
from lit_sender_receiver.agents.rnn_sender import RNNSender
from lit_sender_receiver.base.dataset_base import DatasetBase
from lit_sender_receiver.base.receiver_base import ReceiverOutput
from lit_sender_receiver.base.sender_base import SenderOutput
from lit_sender_receiver.base.game_base import GameBase
from lit_sender_receiver.base.variable_types import Batch, Sample, Source, Target


@pytest.mark.parametrize("cell_type", ["rnn", "gru", "lstm"])
def test_toy_onehot_rnn_reconstruction_game(
    cell_type: Literal["rnn", "gru", "lstm"],
) -> None:

    num_classes = 11
    alphabet_size = 13
    max_message_length = 17
    hidden_size = 127
    batch_size = 31
    lr = 1e-3
    max_epochs = 100

    class OneHotDataset(DatasetBase):
        def __init__(
            self,
            num_classes: int,
        ) -> None:
            super().__init__()
            self.num_classes = num_classes

        def __getitem__(
            self,
            index: int,
        ) -> Sample:
            source = target = F.one_hot(
                torch.as_tensor(index),
                num_classes=self.num_classes,
            ).float()
            return {"source": source, "target": target}

        def __len__(self) -> int:
            return self.num_classes

    class OneHotDataModule(LightningDataModule):
        def __init__(
            self,
            num_classes: int,
            batch_size: int,
        ) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.batch_size = batch_size
            self.onehot_dataset = OneHotDataset(num_classes)

        def train_dataloader(self) -> DataLoader[Batch]:
            return DataLoader(
                self.onehot_dataset,
                batch_size=self.batch_size,
                sampler=RandomSampler(
                    self.onehot_dataset,
                    replacement=True,
                    num_samples=self.batch_size,
                ),
            )

        def val_dataloader(self) -> DataLoader[Batch]:
            return DataLoader(
                self.onehot_dataset,
                batch_size=self.batch_size,
            )

    class OneHotReconstructionGame(GameBase):
        def __init__(
            self,
            sender: RNNSender,
            receiver: RNNReceiver,
            lr: float,
        ) -> None:
            super().__init__(sender, receiver, lr)

        @jaxtyped(typechecker=beartype)
        def loss_function(
            self,
            *,
            source: Source,
            target: Target,
            sender_output: SenderOutput,
            receiver_output: ReceiverOutput,
        ) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
            if source.shape != target.shape or (source != target).any().item():
                raise RuntimeError(
                    "The source and target must be identical in reconstruction games."
                )

            reconstruction_loss = F.cross_entropy(
                receiver_output.prediction, target, reduction="none"
            )

            baseline = reconstruction_loss.mean(dim=0, keepdim=True)

            sender_log_prob = torch.where(
                sender_output.message_mask,
                sender_output.log_prob_seq,
                0,
            ).sum(dim=1)

            sender_loss = (reconstruction_loss - baseline).detach() * sender_log_prob
            receiver_loss = reconstruction_loss

            loss = sender_loss.mean() + receiver_loss.mean()

            return loss, {}

    datamodule = OneHotDataModule(num_classes, batch_size)

    sender = RNNSender(
        source_dim=num_classes,
        alphabet_size=alphabet_size,
        max_message_length=max_message_length,
        cell_type=cell_type,
        hidden_size=hidden_size,
    )

    receiver = RNNReceiver(
        target_dim=num_classes,
        alphabet_size=alphabet_size,
        cell_type=cell_type,
        hidden_size=hidden_size,
    )

    game = OneHotReconstructionGame(sender, receiver, lr)

    trainer = Trainer(max_epochs=max_epochs)

    trainer.fit(game, datamodule=datamodule)
