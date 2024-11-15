import pytest

from data import CustomCharTokenizerSouthParkDataset


@pytest.fixture
def set_up():
    # Mocking the get_all_contained_chars function to control the output
    tokenizer = CustomCharTokenizerSouthParkDataset(['a', 'b', 'c'])

    return tokenizer


def test_initialization(set_up):
    tokenizer = set_up

    # Test char_to_idx mapping
    expected_char_to_idx = {'a': 0, 'b': 1, 'c': 2}
    assert tokenizer.char_to_idx == expected_char_to_idx

    # Test idx_to_char mapping
    expected_idx_to_char = {0: 'a', 1: 'b', 2: 'c'}
    assert tokenizer.idx_to_char == expected_idx_to_char

    # Test vocabulary size
    assert tokenizer.vocab_size == 3


def test_encode(set_up):
    tokenizer = set_up

    # Testing the encode method
    text = 'cabb'
    encoded = tokenizer.encode(text)
    assert encoded.ids == [2, 0, 1, 1]


def test_decode(set_up):
    tokenizer = set_up

    # Testing the decode method using a tensor-like list
    tokens = [2, 0, 1]
    decoded = tokenizer.decode(tokens)
    assert decoded == 'cab'
