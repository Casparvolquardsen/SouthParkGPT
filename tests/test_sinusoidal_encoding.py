import pytest
import torch

from models import SinusoidalPositionalEncoding
from tests.test_config import get_test_config


# Fixtures for testing
@pytest.fixture
def config():
    config = get_test_config()
    config.dropout = 0.1
    config.context_len = 50
    config.d_model = 6
    return config


@pytest.fixture
def model(config):
    return SinusoidalPositionalEncoding(config)


def test_initialization(config):
    model = SinusoidalPositionalEncoding(config)
    assert model.pe is not None, "Positional encoding buffer should be initialized."
    assert model.pe.size() == (
        1,
        config.context_len,
        config.d_model,
    ), "Positional encoding buffer has incorrect size."


def test_forward_shape(config, model):
    batch_size = 2
    seq_len = 8
    embedding_dim = config.d_model

    x = torch.zeros(batch_size, seq_len, embedding_dim)
    output = model(x)

    assert output.size() == (
        batch_size,
        seq_len,
        embedding_dim,
    ), "Output shape is incorrect."


def test_forward_batches(config, model):
    batch_size = 2
    seq_len = 8
    embedding_dim = config.d_model

    x = torch.zeros(batch_size, seq_len, embedding_dim)
    with torch.no_grad():
        model.eval()
        output = model(x)

    assert not torch.equal(x, output), "Output should not be equal to the input."
    assert torch.equal(
        output[0, :, :], output[1, :, :]
    ), "Output should be the same for all batches."


def test_dropout_effect(config, model):
    batch_size = 2
    seq_len = 8
    embedding_dim = config.d_model

    x = torch.zeros(batch_size, seq_len, embedding_dim)
    model.eval()
    output_eval = model(x)
    model.train()
    output_train = model(x)

    assert not torch.equal(
        output_eval, output_train
    ), "Dropout should change the output during training."


# Run the tests
if __name__ == "__main__":
    pytest.main()
