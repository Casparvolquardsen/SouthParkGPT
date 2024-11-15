import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from prettytable import MARKDOWN, PrettyTable
from tqdm import tqdm

from utility.utils import get_tokenizer


@dataclass
class Config:
    tokenizer_type: str


FOLDER_PATH = 'southpark_scripts/all_scripts'
CLEANED_FOLDER_PATH = 'southpark_scripts/all_scripts_cleaned'

file_paths = [
    os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if f.endswith('.csv')
]
cleaned_file_paths = [
    os.path.join(CLEANED_FOLDER_PATH, f)
    for f in os.listdir(CLEANED_FOLDER_PATH)
    if f.endswith('.csv')
]

file_paths.sort()
cleaned_file_paths.sort()

tokenizer_types = [
    'gpt2',
    'bpe-metaspace-256',
    'bpe-metaspace-512',
    'bpe-metaspace-1024',
    'bpe-metaspace-2048',
    'bpe-metaspace-4096',
    'bpe-metaspace-8192',
    'bpe-metaspace-16384',
    'bpe-metaspace-32768',
    'bpe-metaspace-punctuation-256',
    'bpe-metaspace-punctuation-512',
    'bpe-metaspace-punctuation-1024',
    'bpe-metaspace-punctuation-2048',
    'bpe-metaspace-punctuation-4096',
    'bpe-metaspace-punctuation-8192',
    'bpe-metaspace-punctuation-16384',
    'bpe-metaspace-punctuation-32768',
    'bpe-whitespace-punctuation-256',
    'bpe-whitespace-punctuation-512',
    'bpe-whitespace-punctuation-1024',
    'bpe-whitespace-punctuation-2048',
    'bpe-whitespace-punctuation-4096',
    'bpe-whitespace-punctuation-8192',
    'bpe-whitespace-punctuation-16384',
    'bpe-whitespace-punctuation-32768',
]
tokenizers = {
    tokenizer: get_tokenizer(Config(tokenizer_type=tokenizer))
    for tokenizer in tokenizer_types
}

num_chars = 0
num_lines = 0
num_words = 0
num_tokens = {tokenizer: 0 for tokenizer in tokenizer_types}
num_episodes = 0
different_characters = set()

for file_path in tqdm(file_paths):
    with open(file_path, 'r', encoding='utf-8') as file:
        episode_string = file.read()
        num_chars += len(episode_string)
        num_lines += episode_string.count('\n')
        num_words += len(episode_string.split())
        num_episodes += 1
        different_characters.update(set(episode_string))
        for tokenizer in tokenizers:
            num_tokens[tokenizer] += len(tokenizers[tokenizer].encode(episode_string).ids)

table = PrettyTable()
table.field_names = ['Dataset statistic', 'Value']
table.align['Dataset statistic'] = 'l'
table.align['Value'] = 'r'
table.add_row(['Number of characters', f'{num_chars:_}'])
table.add_row(['Number of words', f'{num_words:_}'])
table.add_row(['Number of lines', f'{num_lines:_}'])
table.add_row(['Number of episodes', f'{num_episodes:_}'])
table.set_style(MARKDOWN)
print(table)

tokenizers_table = PrettyTable()
tokenizers_table.field_names = ['Tokenizer', 'Number of tokens']
tokenizers_table.align['Tokenizer'] = 'l'
tokenizers_table.align['Number of tokens'] = 'r'
for tokenizer in tokenizer_types:
    tokenizers_table.add_row([f'{tokenizer}', f'{num_tokens[tokenizer]:_}'])
tokenizers_table.set_style(MARKDOWN)
print(tokenizers_table)

different_characters = sorted(list(different_characters))
print(f'Number of different characters: {len(different_characters)}')
print(f'Different characters: {"".join(different_characters)}')

# Cleaned dataset
num_chars = 0
num_lines = 0
num_words = 0
num_tokens = {tokenizer: 0 for tokenizer in tokenizer_types}
num_episodes = 0
different_characters = set()

for file_path in tqdm(cleaned_file_paths):
    with open(file_path, 'r', encoding='utf-8') as file:
        episode_string = file.read()
        num_chars += len(episode_string)
        num_lines += episode_string.count('\n')
        num_words += len(episode_string.split())
        num_episodes += 1
        different_characters.update(set(episode_string))
        for tokenizer in tokenizers:
            num_tokens[tokenizer] += len(tokenizers[tokenizer].encode(episode_string).ids)

table = PrettyTable()
table.field_names = ['Dataset statistic', 'Value']
table.align['Dataset statistic'] = 'l'
table.align['Value'] = 'r'
table.add_row(['Number of characters', f'{num_chars:_}'])
table.add_row(['Number of words', f'{num_words:_}'])
table.add_row(['Number of lines', f'{num_lines:_}'])
table.add_row(['Number of episodes', f'{num_episodes:_}'])
table.set_style(MARKDOWN)
print(table)

different_characters = sorted(list(different_characters))
print(f'Number of different characters: {len(different_characters)}')
print(f'Different characters: {"".join(different_characters)}')

tokenizers_table = PrettyTable()
tokenizers_table.field_names = ['Tokenizer', 'Number of tokens']
tokenizers_table.align['Tokenizer'] = 'l'
tokenizers_table.align['Number of tokens'] = 'r'
for tokenizer in tokenizer_types:
    tokenizers_table.add_row([f'{tokenizer}', f'{num_tokens[tokenizer]:_}'])
tokenizers_table.set_style(MARKDOWN)
print(tokenizers_table)

# gutenberg dataset
gutenberg_dataset = load_dataset("sedthh/gutenberg_english", split="train")

num_gutenberg_chars = 0
num_gutenberg_lines = 0
num_gutenberg_words = 0

for row in tqdm(gutenberg_dataset):
    num_gutenberg_chars += len(row['TEXT'])
    num_gutenberg_lines += row['TEXT'].count('\n')
    num_gutenberg_words += len(row['TEXT'].split())

gutenberg_table = PrettyTable()
gutenberg_table.field_names = ['Gutenberg dataset statistic', 'Value']
gutenberg_table.align['Gutenberg dataset statistic'] = 'l'
gutenberg_table.align['Value'] = 'r'
gutenberg_table.add_row(['Number of characters', f'{num_gutenberg_chars:_}'])
gutenberg_table.add_row(['Number of words', f'{num_gutenberg_words:_}'])
gutenberg_table.add_row(['Number of lines', f'{num_gutenberg_lines:_}'])
gutenberg_table.set_style(MARKDOWN)
print(gutenberg_table)

tokenizer_dirs = os.listdir('gutenberg_tokenized')
tokenizer_dirs.sort()

gutenberg_tokenizer_table = PrettyTable()
gutenberg_tokenizer_table.field_names = ['Tokenizer', 'Number of tokens']
gutenberg_tokenizer_table.align['Tokenizer'] = 'l'
gutenberg_tokenizer_table.align['Number of tokens'] = 'r'

for tokenizer_dir in tokenizer_dirs:
    num_gutenberg_tokens = 0
    row_files = [
        os.path.join(tokenizer_dir, f)
        for f in os.listdir(os.path.join('gutenberg_tokenized', tokenizer_dir))
        if f.endswith('.pt')
    ]
    for row_file in row_files:
        tokens = torch.load(row_file)
        num_gutenberg_tokens += tokens.shape[0]

    gutenberg_tokenizer_table.add_row([f'{tokenizer_dir}', f'{num_gutenberg_tokens:_}'])

gutenberg_tokenizer_table.set_style(MARKDOWN)
print(gutenberg_tokenizer_table)
