import os

from tokenizers import Tokenizer, decoders, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Metaspace, Punctuation, Sequence, Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import GPT2TokenizerFast

TOKENIZER_DIR = 'custom_tokenizers'


def generate_bpe_metaspace_tokenizer(vocab_size, save_file):
    # Initialize a tokenizer with the BPE model
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    # Initialize the trainer, specifying any training arguments
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=10)

    # Prepare the files with the training data: a list of file paths
    folder_path = 'southpark_scripts/all_scripts'
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.csv')
    ]

    # Train the tokenizer
    tokenizer.train(files, trainer)

    # Save the tokenizer
    tokenizer.save(save_file)


def generate_bpe_metaspace_punctuation_tokenizer(vocab_size, save_file):
    # Initialize a tokenizer with the BPE model
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = Sequence([Metaspace(), Punctuation()])
    tokenizer.decoder = decoders.Metaspace()

    # Initialize the trainer, specifying any training arguments
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=10)

    # Prepare the files with the training data: a list of file paths
    folder_path = 'southpark_scripts/all_scripts'
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.csv')
    ]

    # Train the tokenizer
    tokenizer.train(files, trainer)

    # Save the tokenizer
    tokenizer.save(save_file)


def generate_bpe_whitespce_tokenizer(vocab_size, save_file):
    # Initialize a tokenizer with the BPE model
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
    tokenizer.decoder = decoders.BPEDecoder()

    # Initialize the trainer, specifying any training arguments
    trainer = BpeTrainer(vocab_size=vocab_size - 2)

    # Prepare the files with the training data: a list of file paths
    folder_path = 'southpark_scripts/all_scripts'
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.csv')
    ]

    # Train the tokenizer
    tokenizer.train(files, trainer)

    tokenizer.add_tokens([' ', '\n'])

    # Save the tokenizer
    tokenizer.save(save_file)


class EncodingObject:
    def __init__(self, text, ids):
        self.text = text
        self.ids = ids


class CustomCharTokenizerSouthParkDataset:
    def __init__(self, all_chars=None):
        if all_chars is None:
            all_chars = CustomCharTokenizerSouthParkDataset.get_all_contained_chars()
        self.char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text: str) -> EncodingObject:
        return EncodingObject(text=text, ids=[self.char_to_idx[char] for char in text])

    def decode(self, tokens: list) -> str:
        return ''.join([self.idx_to_char[tokens[idx]] for idx in range(len(tokens))])

    def get_vocab_size(self):
        return self.vocab_size

    # Extract all kinds of characters from the dataset
    @staticmethod
    def get_all_contained_chars(all_scripts_path='southpark_scripts/all_scripts'):
        all_chars = set()
        for file in os.listdir(all_scripts_path):
            if file.endswith('.csv'):
                with open(
                    os.path.join(all_scripts_path, file), 'r', encoding='utf-8'
                ) as f:
                    # previous_size = len(all_chars)
                    text = f.read()
                    all_chars.update(text)
                    # new_chars = set(text)
                    # for char in '¡°½¿É×ØÚàáâäåæçèéëíîñòóôö÷øùúüăĐưΩУЯзийпфхщ،ابحخرسضمنهویảứ–—‘’“”…™∝∠♪':
                    #     if char in new_chars:
                    #         print(f"{char} in {file}")
                    # print(f"New chars: {len(all_chars) - previous_size}, episode: {file}")

        # set to string
        all_chars = list(all_chars)
        all_chars.sort()
        all_chars_str = ''.join(all_chars)
        return all_chars_str


class GPT2Tokenizer:
    def __init__(self):
        self._tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    def encode(self, text: str) -> EncodingObject:
        return EncodingObject(text=text, ids=self._tokenizer.encode(text))

    def decode(self, tokens: list) -> str:
        return self._tokenizer.decode(tokens)

    def get_vocab_size(self):
        return self._tokenizer.vocab_size


if __name__ == '__main__':
    for vocab_size in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        save_file = os.path.join(
            TOKENIZER_DIR, f"bpe_metaspace_tokenizer_{vocab_size}.json"
        )
        generate_bpe_metaspace_tokenizer(vocab_size, save_file)
        print(f"Tokenizer with vocab size {vocab_size} saved to {save_file}")

    for vocab_size in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        save_file = os.path.join(
            TOKENIZER_DIR, f"bpe_whitespace_punctuation_tokenizer_{vocab_size}.json"
        )
        generate_bpe_whitespce_tokenizer(vocab_size, save_file)
        print(f"Tokenizer with vocab size {vocab_size} saved to {save_file}")

    for vocab_size in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        save_file = os.path.join(
            TOKENIZER_DIR, f"bpe_metaspace_punctuation_tokenizer_{vocab_size}.json"
        )
        generate_bpe_metaspace_punctuation_tokenizer(vocab_size, save_file)
        print(f"Tokenizer with vocab size {vocab_size} saved to {save_file}")
