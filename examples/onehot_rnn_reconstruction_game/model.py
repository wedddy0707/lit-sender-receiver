from typing import Literal

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from lit_sender_receiver.agents.rnn_receiver import RNNReceiver
from lit_sender_receiver.agents.rnn_sender import RNNSender
from lit_sender_receiver.base.receiver_base import ReceiverOutput
from lit_sender_receiver.base.sender_base import SenderOutput
from lit_sender_receiver.base.game_base import GameBase
from lit_sender_receiver.base.variable_types import Source, Target


class OneHotRNNReconstructionGame(GameBase):
    def __init__(
        self,
        num_classes: int,
        alphabet_size: int,
        max_message_length: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        hidden_size: int,
        sender_entropy_regularizer_coeff: float,
        lr: float,
    ) -> None:
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

        super().__init__(sender, receiver, lr)

        self.sender_entropy_regularizer_coeff = sender_entropy_regularizer_coeff

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
            sender_output.message_mask, sender_output.log_prob_seq, 0
        ).sum(dim=1)

        sender_entropy_regularizer = torch.where(
            sender_output.message_mask, sender_output.entropy_seq, 0.0
        ).sum(dim=1) / sender_output.message_mask.float().sum(dim=1)

        sender_loss = (
            (reconstruction_loss - baseline).detach() * sender_log_prob
            - sender_entropy_regularizer * self.sender_entropy_regularizer_coeff
        )
        receiver_loss = reconstruction_loss

        total_loss = sender_loss.mean() + receiver_loss.mean()

        with torch.no_grad():
            acc = (
                (receiver_output.prediction.argmax(dim=-1) == target.argmax(dim=-1))
                .float()
                .mean()
            )
            message_length = sender_output.message_mask.long().sum(dim=-1)

        return total_loss, {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "acc": acc,
            "message_length": message_length,
            "sender_entropy_regularizer": sender_entropy_regularizer,
        }
