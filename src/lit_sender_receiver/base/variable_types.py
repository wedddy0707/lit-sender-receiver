from typing import TypedDict

from jaxtyping import Bool, Float, Integer
from torch import Tensor

BatchedFloatScalar = Float[Tensor, " batch_size"]
BatchedIntegerScalar = Integer[Tensor, " batch_size"]

SourceSample = Float[Tensor, " source_dim"]
TargetSample = Float[Tensor, " target_dim"]

Source = Float[Tensor, "batch_size source_dim"]
Target = Float[Tensor, "batch_size target_dim"]
Prediction = Float[Tensor, "batch_size prediction_dim"]

Message = Integer[Tensor, "batch_size seq_len"]
MessageMask = Bool[Tensor, "batch_size seq_len"]
MessageLength = BatchedIntegerScalar

LogProbSeq = Float[Tensor, "batch_size seq_len"]
EntropySeq = Float[Tensor, "batch_size seq_len"]


class Sample(TypedDict):
    source: SourceSample
    target: TargetSample


class Batch(TypedDict):
    source: Source
    target: Target
