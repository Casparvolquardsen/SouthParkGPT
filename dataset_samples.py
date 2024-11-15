import random
from dataclasses import dataclass

from utility import generate
from utility.utils import get_dataset, get_model, get_tokenizer


@dataclass
class Config:
    vocab_size: int = None
    context_len: int = 256
    num_samples: int = None
    tokenizer_type: str = 'bpe-metaspace-punctuation-512'
    dataset: str = 'cleaned-southpark'

    architecture: str = 'decoder-transformer'
    bias: bool = True
    d_model: int = 384
    d_feedforward: int = 1536
    dropout: float = 0.1
    num_heads: int = 6
    num_layers: int = 6
    positional_encoding: str = 'learned'
    device: str = 'mps'
    pretrained_model: str = (
        'models_params/examples/example2_finetuned_southpark_decoder-transformer.pth'
    )


config = Config()
tokenizer = get_tokenizer(config)
config.vocab_size = tokenizer.get_vocab_size()
dataset = get_dataset(tokenizer, config)
model = get_model(config)

for i in range(10):
    src, _ = dataset[random.randint(0, len(dataset))]
    text = tokenizer.decode(src.tolist())
    print(text)
    print('---')

print('Generated samples:')
for i in range(10):
    src, _ = dataset[random.randint(0, len(dataset))]
    text = generate(
        model=model,
        tokens=src,
        seq_len=config.context_len,
        tokenizer=tokenizer,
        max_len_generate=256,
        temperature=0.7,
        device=config.device,
    )
    print(text)
    print('---')
