from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch.nn import Module

from .variable_types import EntropySeq, LogProbSeq, Message, MessageMask, Source


@dataclass
class SenderOutput:
    message: Message
    message_mask: MessageMask
    log_prob_seq: LogProbSeq
    entropy_seq: EntropySeq


class SenderBase(Module, ABC):
    @abstractmethod
    def forward(
        self,
        source: Source,
    ) -> SenderOutput:
        raise NotImplementedError()
