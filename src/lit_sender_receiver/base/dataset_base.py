from torch.utils.data import Dataset

from .variable_types import Sample


class DatasetBase(Dataset[Sample]):
    pass
