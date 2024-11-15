import argparse

import yaml


def get_args():
    parser = argparse.ArgumentParser(description='Southpark-LM')

    parser.add_argument(
        '--architecture',
        type=str,
        default='decoder-transformer',
        help='Model architecture: decoder-transformer | rnn-baseline',
    )

    parser.add_argument(
        '--batch_size', type=int, default=48, help='Batch size for training'
    )

    parser.add_argument(
        '--beam_search', type=bool, default=False, help='Use beam search during inference'
    )

    parser.add_argument(
        '--bias', type=bool, default=True, help='Use bias in linear layers'
    )

    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        help='Beta1 value for Adam and AdamW optimizers',
    )

    parser.add_argument(
        '--beta2',
        type=float,
        default=0.95,
        help='Beta2 value for Adam and AdamW optimizers',
    )

    parser.add_argument(
        '--config', type=str, help='Path to the configuration YAML file', default=None
    )

    parser.add_argument(
        '--context_len',
        type=int,
        default=256,
        help='Length of the input sequence (context window)',
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='southpark',
        help='Dataset for training: southpark | gutenberg',
    )

    parser.add_argument(
        '--d_feedforward',
        type=int,
        default=1536,
        help='Dimension of the feedforward network in the transformer blocks',
    )

    parser.add_argument(
        '--d_model', type=int, default=384, help='Dimension of the embeddings'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='accelerated',
        help='Device to use for training: mps | cuda | cpu | accelerated',
    )

    parser.add_argument(
        '--dropout', type=float, default=0.1, help='Dropout rate while training'
    )

    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')

    parser.add_argument(
        '--evaluation_interval',
        type=int,
        default=1000,
        help='Interval for model evaluation',
    )

    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=10,
        help='Number of gradient accumulation steps before updating the model',
    )

    parser.add_argument(
        '--gradient_clip_value', type=float, default=1.0, help='Gradient clipping value'
    )

    parser.add_argument(
        '--log_text_samples',
        type=bool,
        default=True,
        help='Log generated text samples to Weights & Biases',
    )

    parser.add_argument(
        '--log_wandb',
        type=bool,
        default=False,
        help='Log training metrics to Weights & Biases',
    )

    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')

    parser.add_argument(
        '--lr_decay_iters', type=int, default=9000, help='Learning rate decay iterations'
    )

    parser.add_argument(
        '--max_len_generate',
        type=int,
        default=300,
        help='Maximum number of tokens of generated text',
    )

    parser.add_argument(
        '--min_lr',
        type=float,
        default=6e-5,
        help='Minimum learning rate (used in the scheduler)',
    )

    parser.add_argument(
        '--num_heads', type=int, default=6, help='Number of attention heads'
    )

    parser.add_argument(
        '--num_layers', type=int, default=6, help='Number of layers in the model'
    )

    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to use (None for all samples)',
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        help='Optimizer type: adam | adamw | rmsprop',
    )

    parser.add_argument(
        '--positional_encoding',
        type=str,
        default='learned',
        help='Type of positional encoding: sinusoidal | learned',
    )

    parser.add_argument(
        '--pretrained_model', type=str, default='', help='Path to the pretrained model'
    )

    parser.add_argument(
        '--prompt',
        type=str,
        default='Cartman,',
        help='Prompt for text generation. Only used in generate.py.',
    )

    parser.add_argument(
        '--prompt_file',
        type=str,
        default=None,
        help='Path to a text file containing a prompt for text generation. Only used in generate.py.',
    )

    parser.add_argument(
        '--save_model_every_epoch',
        type=bool,
        default=False,
        help='Save model checkpoints after every epoch',
    )

    parser.add_argument(
        '--scheduler',
        type=str,
        default='cosine',
        help='Learning rate scheduler: noam | cosine | constant | none',
    )

    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--split_ratios',
        type=str,
        default='90-5-5',
        help='Data split ratios for training, validation, and testing: 90-5-5 | 80-10-10 | 99-0.5-0.5 | 99.9-0.05-0.05',
    )

    parser.add_argument(
        '--start_text',
        type=str,
        default='',
        help='Initial text prompt for text generation',
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for sampling during inference',
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        default='bpe-metaspace-punctuation-512',
        help='Tokenizer type: char-level | gpt2 | bpe-metaspace-XXXX | bpe-metaspace-punctuation-XXXX | bpe-whitespace-punctuation-XXXX',
    )

    parser.add_argument(
        '--use_attn_mask',
        type=bool,
        default=True,
        help='Use attention masking during training',
    )

    parser.add_argument(
        '--use_gradient_clipping',
        type=bool,
        default=True,
        help='Enable gradient clipping',
    )

    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=500,
        help='Number of warmup steps for the scheduler',
    )

    parser.add_argument(
        '--weight_decay', type=float, default=1e-1, help='Weight decay rate'
    )

    args = parser.parse_args()

    # Load YAML config if provided
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)

        # Overwrite default args with YAML config
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    assert args.architecture.lower() in ['decoder-transformer', 'rnn']

    assert args.tokenizer.lower() in [
        'char-level',
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

    assert args.optimizer.lower() in ['adam', 'adamw', 'rmsprop']

    assert args.scheduler is None or args.scheduler.lower() in [
        'noam',
        'cosine',
        'constant',
        'none',
    ]

    assert args.device.lower() in ['mps', 'cuda', 'cpu', 'accelerated']

    assert args.positional_encoding.lower() in ['sinusoidal', 'learned']

    assert args.dataset.lower() in [
        'southpark',
        'cleaned-southpark',
        'gutenberg',
        'shuffled-gutenberg',
    ]

    assert args.split_ratios in ['90-5-5', '80-10-10', '99-0.5-0.5', '99.9-0.05-0.05']

    return args
