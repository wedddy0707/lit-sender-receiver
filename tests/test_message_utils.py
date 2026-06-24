import pytest
import torch

from lit_sender_receiver.utils.message_utils import compute_message_mask


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_compute_message_mask(device: torch.device):
    message = torch.as_tensor(
        (
            (0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0),
            (1, 2, 0, 0, 0, 0, 0),
            (1, 2, 3, 0, 0, 0, 0),
            (1, 2, 3, 4, 0, 0, 0),
            (1, 2, 3, 4, 5, 0, 0),
            (1, 2, 3, 4, 5, 6, 0),
            (1, 2, 3, 4, 5, 6, 7),
            (0, 0, 0, 0, 0, 0, 1),
            (0, 0, 0, 0, 0, 2, 1),
            (0, 0, 0, 0, 3, 2, 1),
            (0, 0, 0, 4, 3, 2, 1),
            (0, 0, 5, 4, 3, 2, 1),
            (0, 6, 5, 4, 3, 2, 1),
        ),
        dtype=torch.long,
        device=device,
    )
    message_mask = torch.as_tensor(
        (
            (1, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0),
            (1, 1, 1, 0, 0, 0, 0),
            (1, 1, 1, 1, 0, 0, 0),
            (1, 1, 1, 1, 1, 0, 0),
            (1, 1, 1, 1, 1, 1, 0),
            (1, 1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1, 1),
            (1, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0),
        ),
        dtype=torch.bool,
        device=device,
    )

    assert (compute_message_mask(message) == message_mask).all().item()
