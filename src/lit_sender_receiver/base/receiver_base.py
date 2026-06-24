from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch.nn import Module

from .variable_types import Message, MessageMask, Prediction


@dataclass
class ReceiverOutput:
    prediction: Prediction


class ReceiverBase(Module, ABC):
    @abstractmethod
    def forward(
        self,
        message: Message,
        message_mask: MessageMask,
    ) -> ReceiverOutput:
        raise NotImplementedError()
