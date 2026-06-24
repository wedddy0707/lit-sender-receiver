from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from beartype import beartype
from jaxtyping import Float, jaxtyped
from lightning import LightningModule
from torch import Tensor
from torch.optim import Adam, Optimizer

from .receiver_base import ReceiverBase, ReceiverOutput
from .sender_base import SenderBase, SenderOutput
from .variable_types import Source, Target, Batch


@dataclass
class GameOutput:
    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]
    sender_output: SenderOutput
    receiver_output: ReceiverOutput


class GameBase(LightningModule, ABC):
    sender: SenderBase
    receiver: ReceiverBase
    lr: float

    def __init__(
        self,
        sender: SenderBase,
        receiver: ReceiverBase,
        lr: float,
    ) -> None:
        super().__init__()

        self.sender = sender
        self.receiver = receiver
        self.lr = lr

    @abstractmethod
    def loss_function(
        self,
        *,
        source: Source,
        target: Target,
        sender_output: SenderOutput,
        receiver_output: ReceiverOutput,
    ) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
        raise NotImplementedError()

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        *,
        source: Source,
        target: Target,
    ) -> GameOutput:
        sender_output = self.sender.forward(
            source=source,
        )
        receiver_output = self.receiver.forward(
            message=sender_output.message,
            message_mask=sender_output.message_mask,
        )
        loss, metrics = self.loss_function(
            source=source,
            target=target,
            sender_output=sender_output,
            receiver_output=receiver_output,
        )
        return GameOutput(
            loss=loss,
            metrics=metrics,
            sender_output=sender_output,
            receiver_output=receiver_output,
        )

    @jaxtyped(typechecker=beartype)
    def training_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> Float[Tensor, ""]:
        game_output = self.forward(source=batch["source"], target=batch["target"])

        for k, v in game_output.metrics.items():
            self.log("train/" + k, v.float().mean())

        return game_output.loss

    @jaxtyped(typechecker=beartype)
    def validation_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> Float[Tensor, ""]:
        game_output = self.forward(source=batch["source"], target=batch["target"])

        for k, v in game_output.metrics.items():
            self.log("val/" + k, v.float().mean())

        return game_output.loss

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.lr)
