import logging
import os
import random
from multiprocessing import Pool

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from utility import Config

logger = logging.getLogger(__name__)


class SouthParkScriptsDataset(Dataset):
    def __init__(
        self, tokenizer, config: Config, folder_path='southpark_scripts/all_scripts'
    ):
        self.file_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.csv')
        ]
        self.file_paths.sort()
        self.episodes = []
        self.context_len = config.context_len
        self.num_samples = 0

        # Read each file and collect lines
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                episode_string = file.read()
                episode_tokens = torch.tensor(
                    tokenizer.encode(episode_string).ids, dtype=torch.long
                )
                episode_name = os.path.basename(file_path).split('.')[0]
                episode_start_sample = self.num_samples
                self.num_samples += (
                    episode_tokens.shape[0] - self.context_len
                )  # -1 because we predict the next
                # token

                self.episodes.append(
                    (episode_name, episode_start_sample, self.num_samples, episode_tokens)
                )

                if (
                    config.num_samples is not None
                    and self.num_samples >= config.num_samples
                ):
                    break

        if config.num_samples is not None:
            self.num_samples = config.num_samples

    def __len__(self):
        return self.num_samples

    def get_episode_by_sample_index(self, sample_index):
        return self._get_fitting_episode(sample_index)[2]

    def _get_fitting_episode(self, idx):
        episode_idx = 0

        for episode_name, start, end, tokens in self.episodes:
            if start <= idx < end:
                return idx - start, tokens, episode_idx
            episode_idx += 1
        return None, None, None

    def __getitem__(self, index):
        episode_idx, episode_tokens, _ = self._get_fitting_episode(index)
        if episode_idx is not None:
            src = episode_tokens[episode_idx : episode_idx + self.context_len]
            tgt = episode_tokens[episode_idx + 1 : episode_idx + self.context_len + 1]
            return src, tgt
        return None, None


# 5_431_540_519 samples with gpt2 tokenizer
class GutenbergDataset(Dataset):
    def __init__(self, tokenizer, config):
        tokenized_dataset_dir = self._store_tokenized_dataset_on_disk(tokenizer, config)

        self.context_len = config.context_len

        self.rows = []
        current_num_samples = 0

        # get all tokenized files (ending with .pt)
        row_files = [
            os.path.join(tokenized_dataset_dir, f)
            for f in os.listdir(tokenized_dataset_dir)
            if f.endswith('.pt')
        ]

        logger.info('Preparing Gutenberg dataset...')
        for row_file in tqdm(row_files):
            if (
                config.num_samples is not None
                and current_num_samples >= config.num_samples
            ):
                break

            tokens = torch.load(row_file)
            row_start_sample = current_num_samples
            current_num_samples += tokens.shape[0] - self.context_len
            self.rows.append((row_start_sample, current_num_samples, row_file))

        # Calculate the number of context length samples
        self.num_samples = (
            current_num_samples if config.num_samples is None else config.num_samples
        )

    # Tokenize the Gutenberg dataset and store it on disk as torch tensors
    # This is done to avoid loading the whole dataset into memory at once
    # and also to avoid tokenizing the same text multiple times
    # Using 10 processes on a 10-core M1 Max MacBook Pro, this takes about 22 minutes for gpt2 tokenizer
    # This takes around 22 GB of disk space
    @staticmethod
    def _store_tokenized_dataset_on_disk(
        tokenizer, config, dataset_dir='data/gutenberg_tokenized'
    ):
        tokenized_dataset_dir = os.path.join(dataset_dir, config.tokenizer_type)

        if os.path.exists(tokenized_dataset_dir):
            return tokenized_dataset_dir
        else:
            os.makedirs(tokenized_dataset_dir, exist_ok=True)

        logger.info("Loading dataset...")
        dataset = load_dataset("sedthh/gutenberg_english", split="train")

        row_indices = range(len(dataset))

        with Pool(int(os.getenv('NUM_CORES', 10))) as p:
            p.starmap(
                GutenbergDataset._tokenize_row,
                zip(
                    row_indices,
                    [dataset] * len(row_indices),
                    [tokenizer] * len(row_indices),
                    [tokenized_dataset_dir] * len(row_indices),
                ),
            )

        return tokenized_dataset_dir

    @staticmethod
    def _tokenize_row(row_idx, dataset, tokenizer, tokenized_dataset_dir):
        text = dataset[row_idx]['TEXT']
        tokens = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
        torch.save(tokens, os.path.join(tokenized_dataset_dir, f"row_{row_idx:05d}.pt"))

    def __len__(self):
        return self.num_samples

    def _get_fitting_row(self, idx):
        for start, end, tokens_file in self.rows:
            if start <= idx < end:
                return idx - start, torch.load(tokens_file)
        return None, None

    def __getitem__(self, index):
        idx_in_row, row_tokens = self._get_fitting_row(index)
        if idx_in_row is not None:
            src = row_tokens[idx_in_row : idx_in_row + self.context_len]
            tgt = row_tokens[idx_in_row + 1 : idx_in_row + self.context_len + 1]
            return src, tgt
        return None, None


class ShuffledGutenbergDataset(Dataset):
    def __init__(self, tokenizer, config):
        self.num_samples = config.num_samples
        self.random = random.Random('gutenberg-shuffle-seed')

        config.num_samples = None  # to load all samples
        self.gutenberg_dataset = GutenbergDataset(tokenizer, config)
        config.num_samples = self.num_samples  # restore the original value

        # Shuffle the dataset
        logger.info('Shuffled Gutenberg dataset')

    def __len__(self):
        return self.num_samples

    def _get_shuffled_index(self, idx):
        # Generate a pseudo-random number in the range [0, total_elements)
        shuffled_idx = self.random.randint(0, self.num_samples - 1)
        return shuffled_idx

    def __getitem__(self, index):
        return self.gutenberg_dataset[self._get_shuffled_index(index)]


class DatasetSplit(Dataset):
    def __init__(self, dataset, start_idx, end_idx):
        self.dataset = dataset
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_samples = end_idx - start_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.dataset[self.start_idx + index]

    def get_episode_by_sample_index(self, sample_index):
        return self.dataset.get_episode_by_sample_index(sample_index)
