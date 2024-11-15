import torch
import torch.nn.functional as F

from models import generate_look_ahead_mask


def test_generate_look_ahead_mask():
    # Testing the generate_look_ahead_mask function
    size = 5
    mask = generate_look_ahead_mask(size)
    expected_mask = (
        torch.tensor(
            [
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )
        == 1
    )

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            assert mask[i, j] == expected_mask[i, j]


def test_softmax_use_mask():
    mask = generate_look_ahead_mask(5)

    mask = torch.zeros_like(mask, dtype=float).masked_fill_(mask, float("-inf"))

    attention_scores = torch.randn(32, 5, 5)
    attn_output_weights = F.softmax(attention_scores + mask, dim=-1)
    for i in range(attn_output_weights.shape[0]):
        for j in range(attn_output_weights.shape[1]):
            for k in range(attn_output_weights.shape[2]):
                if j < k:
                    assert attn_output_weights[i, j, k] == 0
