from types import SimpleNamespace

import pytest
import torch
from datasets import load_dataset

from data import GutenbergDataset


class MockTokenizer:
    def __init__(self):
        self.pad_token = '<PAD>'
        self.eos_token = '<EOS>'
        self.pad_token_id = 0

    def __call__(self, text, return_tensors, padding, truncation):
        return {
            'input_ids': torch.tensor(
                list(range(len(text)))
            )  # Mocked token ids for simplicity
        }

    def encode(self, text):
        return SimpleNamespace(ids=list(range(len(text))))  # Mocked encoding

    def decode(self, tokens):
        return ''.join([chr(token) for token in tokens])  # Mocked decoding


class MockConfig:
    def __init__(self, context_len, pretraining_dataset_fraction):
        self.context_len = context_len
        self.pretraining_dataset_fraction = pretraining_dataset_fraction


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.fixture
def config():
    return MockConfig(context_len=5, pretraining_dataset_fraction=0.001)


@pytest.fixture
def gutenberg_english():
    return load_dataset("sedthh/gutenberg_english", split="train")


def test_gutenberg_dataset_initialization(tokenizer, config):
    dataset = GutenbergDataset(tokenizer=tokenizer, config=config)
    assert len(dataset) > 0
    assert isinstance(dataset.tokenizer, MockTokenizer)
    assert dataset.context_len == config.context_len


def test_gutenberg_dataset_length(tokenizer, config, gutenberg_english):
    dataset = GutenbergDataset(tokenizer=tokenizer, config=config)
    expected_episodes = int(
        config.pretraining_dataset_fraction * (gutenberg_english).num_rows
    )
    assert dataset.num_episodes == expected_episodes
    expected_length = (len(dataset.tokens) - 1) // config.context_len
    assert len(dataset) == expected_length


def test_gutenberg_dataset_getitem(tokenizer, config):
    dataset = GutenbergDataset(tokenizer=tokenizer, config=config)

    # don't need to check everything but last 5% are where i'd imagine bugs
    for idx in range(len(dataset) // 20, -1):
        input_ids, target_ids = dataset[idx]
        assert len(input_ids) == config.context_len
        assert len(target_ids) == config.context_len


def test_gutenberg_dataset_seperate_texts_padding(tokenizer, config, gutenberg_english):
    dataset = GutenbergDataset(tokenizer=tokenizer, config=config)

    # length of first text in dataset
    length_first_text = len(gutenberg_english[0]["TEXT"])
    pos_end_first_text = (length_first_text - 1) * config.context_len
    end_batch = dataset[pos_end_first_text]
    x_ids, y_ids = end_batch

    full_pad = torch.full((config.context_len,), dataset.tokenizer.pad_token_id)

    # assert all y_ids to be pad token (index 0)
    assert torch.equal(x_ids[1:], full_pad[1:])
    assert torch.equal(y_ids, full_pad)


def test_gutenberg_dataset_encode(tokenizer, config):
    dataset = GutenbergDataset(tokenizer=tokenizer, config=config)
    text = "Cartman"
    encoding = dataset.encode(text)

    assert encoding.text == text
    assert encoding.ids == list(range(len(text)))  # Mocked ids


def test_gutenberg_dataset_decode(tokenizer, config):
    dataset = GutenbergDataset(tokenizer=tokenizer, config=config)
    tokens = [97, 98, 99]  # These would be 'a', 'b', 'c'
    decoded_text = dataset.decode(tokens)
    assert decoded_text == 'abc'
