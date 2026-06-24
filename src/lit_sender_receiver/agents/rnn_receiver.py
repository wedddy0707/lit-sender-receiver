from typing import Literal

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from torch.nn import GRUCell, Linear, LSTMCell, RNNCell
from torch.nn.functional import one_hot

from ..base.receiver_base import ReceiverBase, ReceiverOutput
from ..base.variable_types import Message, MessageMask


class RNNReceiver(ReceiverBase):
    def __init__(
        self,
        *,
        target_dim: int,
        alphabet_size: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        hidden_size: int,
    ) -> None:
        super().__init__()

        match cell_type:
            case "rnn":
                cell_class = RNNCell
            case "gru":
                cell_class = GRUCell
            case "lstm":
                cell_class = LSTMCell
            case _:
                raise ValueError(f"Unknown cell type {cell_type}")

        self.cell = cell_class(alphabet_size + 1, hidden_size)
        self.state_to_prediction = Linear(hidden_size, target_dim)

        self.alphabet_size = self.bos_id = alphabet_size
        self.hidden_size = hidden_size

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        message: Message,
        message_mask: MessageMask,
    ) -> ReceiverOutput:

        message = torch.cat(
            [torch.full_like(message[:, 0:1], self.bos_id), message], dim=1
        )
        message_mask = torch.cat(
            [torch.ones_like(message_mask[:, 0:1]), message_mask], dim=1
        )

        h = c = torch.zeros((message.shape[0], self.hidden_size), device=message.device)

        for symbol, mask in zip(message.unbind(dim=1), message_mask.unbind(dim=1)):
            symbol_onehot = one_hot(symbol, self.alphabet_size + 1).float()
            mask_expanded = mask.unsqueeze(-1).expand_as(h)

            match self.cell:
                case RNNCell() | GRUCell():
                    h = torch.where(
                        mask_expanded,
                        self.cell.forward(symbol_onehot, h),
                        h,
                    )
                case LSTMCell():
                    h_, c_ = self.cell.forward(symbol_onehot, (h, c))
                    h = torch.where(mask_expanded, h_, h)
                    c = torch.where(mask_expanded, c_, c)

        prediction = self.state_to_prediction.forward(h)

        return ReceiverOutput(
            prediction=prediction,
        )
