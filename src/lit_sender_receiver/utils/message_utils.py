from beartype import beartype
from jaxtyping import jaxtyped

from ..base.variable_types import Message, MessageMask


@jaxtyped(typechecker=beartype)
def compute_message_mask(message: Message) -> MessageMask:
    is_eos = (message == 0).long()
    return is_eos.cumsum(dim=-1) - is_eos == 0
