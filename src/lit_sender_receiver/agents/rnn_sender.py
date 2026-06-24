from typing import Literal

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import GRUCell, Linear, LSTMCell, RNNCell
from torch.nn.functional import one_hot

from ..base.sender_base import SenderBase, SenderOutput
from ..base.variable_types import BatchedFloatScalar, BatchedIntegerScalar, Source
from ..utils.message_utils import compute_message_mask


class RNNSender(SenderBase):
    def __init__(
        self,
        *,
        source_dim: int,
        alphabet_size: int,
        max_message_length: int,
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

        self.source_to_state = Linear(source_dim, hidden_size)
        self.state_to_logits = Linear(hidden_size, alphabet_size)

        self.alphabet_size = self.bos_id = alphabet_size
        self.max_message_length = max_message_length

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        source: Source,
    ) -> SenderOutput:

        h = self.source_to_state.forward(source)
        c = torch.zeros_like(h)
        symbol = torch.full(
            (h.shape[0],), self.bos_id, dtype=torch.long, device=h.device
        )

        symbol_list: list[BatchedIntegerScalar] = []
        log_prob_list: list[BatchedFloatScalar] = []
        entropy_list: list[BatchedFloatScalar] = []

        for _ in range(self.max_message_length - 1):
            symbol_onehot = one_hot(symbol, self.alphabet_size + 1).float()

            match self.cell:
                case RNNCell() | GRUCell():
                    h = self.cell.forward(symbol_onehot, h)
                case LSTMCell():
                    h, c = self.cell.forward(symbol_onehot, (h, c))

            logits = self.state_to_logits.forward(h)
            categorical = Categorical(logits=logits)

            if self.training:
                symbol = categorical.sample()
            else:
                symbol = logits.argmax(dim=1)

            symbol_list.append(symbol)
            log_prob_list.append(categorical.log_prob(symbol))
            entropy_list.append(categorical.entropy())

        symbol_list.append(torch.zeros_like(symbol_list[-1]))
        log_prob_list.append(torch.zeros_like(log_prob_list[-1]))
        entropy_list.append(torch.zeros_like(entropy_list[-1]))

        message = torch.stack(symbol_list, dim=1)
        message_mask = compute_message_mask(message)
        message = torch.where(message_mask, message, 0)

        log_prob_seq = torch.where(message_mask, torch.stack(log_prob_list, dim=1), 0.0)
        entropy_seq = torch.where(message_mask, torch.stack(entropy_list, dim=1), 0.0)

        return SenderOutput(
            message=message,
            message_mask=message_mask,
            log_prob_seq=log_prob_seq,
            entropy_seq=entropy_seq,
        )
