import os
import random
import shutil

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, Dataset

import wandb
from data import (
    CustomCharTokenizerSouthParkDataset,
    DatasetSplit,
    GPT2Tokenizer,
    GutenbergDataset,
    ShuffledGutenbergDataset,
    SouthParkScriptsDataset,
)
from evaluate import evaluate_bleu_score
from models import (
    ConstantScheduler,
    CosineRateScheduler,
    DecoderTransformer,
    NoamLR,
    RNNBaseline,
    configure_adamw_optimizer,
)
from utility import Config, beam_search_generate, generate


def model_size(model: torch.nn):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    return size_model


def split(data_set, split_ratios):
    assert sum(split_ratios) - 1.0 < 1e-6, "Split ratios should sum to 1.0"
    num_samples = len(data_set)
    splitted_data_sets = []

    start_idx = 0
    for ratio in split_ratios:
        end_idx = start_idx + int(ratio * num_samples)
        splitted_data_sets.append(DatasetSplit(data_set, start_idx, end_idx))
        start_idx = end_idx

    return splitted_data_sets


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)  # maybe it will be used internally in torch


def choose_prompt_samples(
    num_prompts: int, dataset, generation_length: int, is_southpark_dataset: bool
):
    prompt_samples = []
    references = []

    for i in range(num_prompts):
        # if sample_index < len(dataset):
        sample_index = (i * 10_000) % len(dataset)

        if is_southpark_dataset:
            # only take indices where a reference can be extracted
            while not dataset.get_episode_by_sample_index(
                sample_index
            ) == dataset.get_episode_by_sample_index(sample_index + generation_length):
                sample_index = (sample_index + generation_length) % len(dataset)

        prompt_sample = dataset[sample_index][0]
        prompt_samples.append(prompt_sample)

        if is_southpark_dataset:
            reference_index = sample_index + prompt_sample.size(0)
            reference = dataset[reference_index][0]
            references.append(reference[:generation_length])

    prompt_samples = torch.stack(prompt_samples)
    return (prompt_samples, references)


def log_text_evaluation(
    data_label, num_prompts, model, tokenizer, dataset, step, config: Config
):
    is_southpark_dataset = (
        config.dataset == 'southpark' or config.dataset == 'cleaned-southpark'
    )
    prompt_samples, references = choose_prompt_samples(
        num_prompts, dataset, config.max_len_generate, is_southpark_dataset
    )

    rows = []
    bleu_scores_sum = 0

    for i in range(num_prompts):
        prompt = prompt_samples[i]
        decoded_prompt = tokenizer.decode(prompt.tolist())

        # maximally generate context_len tokens since bleu score reference is not longer than that
        max_len_generate = min(config.max_len_generate, config.context_len)

        if config.beam_search:
            generated_text = beam_search_generate(
                model,
                prompt,
                config.context_len,
                tokenizer,
                max_len_generate=max_len_generate,
                temperature=config.temperature,
                device=config.device,
            )
        else:
            generated_text = generate(
                model,
                prompt,
                config.context_len,
                tokenizer,
                max_len_generate=max_len_generate,
                temperature=config.temperature,
                device=config.device,
            )

        prediction = generated_text

        html_bleu_score_th = ""
        html_bleu_score_td = ""

        if is_southpark_dataset:
            reference = tokenizer.decode(references[i].tolist())
            bleu_score = evaluate_bleu_score(prediction, reference).float().item()
            bleu_scores_sum += bleu_score

            html_bleu_score_th = (
                "<th style='padding: 10px; border: 1px solid black;'>Bleu Score</th>"
            )

            html_bleu_score_td = (
                f"<td style='padding: 10px; border: 1px solid black;'>{bleu_score}</td>"
            )

        rows.append(
            f"<tr><td style='padding: 10px; border: 1px solid black;'><pre style='margin: 0;'>{decoded_prompt}</pre></td><td style='padding: 10px; border: 1px solid black;'><pre style='margin: 0;'>{generated_text}</pre></td>{html_bleu_score_td}</tr>"
        )

    # Create HTML table
    table_html = f"<table style='border-collapse: collapse;'><tr><th style='padding: 10px; border: 1px solid black;'>Prompt</th><th style='padding: 10px; border: 1px solid black;'>Generated Text</th>{html_bleu_score_th}</tr>{''.join(rows)}</table>"
    # print(table_html)
    avg_bleu_score = bleu_scores_sum / num_prompts

    if is_southpark_dataset:
        wandb.log(
            {
                f"{data_label}/generation_samples": wandb.Html(table_html, inject=False),
                f"{data_label}/generation_avg_bleu_score": avg_bleu_score,
            },
            step=step,
        )
    else:
        wandb.log(
            {f"{data_label}/generation_samples": wandb.Html(table_html, inject=False)},
            step=step,
        )


def get_tokenizer(config: Config):
    if config.tokenizer_type == 'char-level':
        tokenizer = CustomCharTokenizerSouthParkDataset()
    elif config.tokenizer_type == 'gpt2':
        tokenizer = GPT2Tokenizer()
    elif config.tokenizer_type == 'bpe-metaspace-256':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_tokenizer_256.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-512':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_tokenizer_512.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-1024':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_tokenizer_1024.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-2048':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_tokenizer_2048.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-4096':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_tokenizer_4096.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-8192':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_tokenizer_8192.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-16384':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_tokenizer_16384.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-32768':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_tokenizer_32768.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-punctuation-256':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_punctuation_tokenizer_256.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-punctuation-512':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_punctuation_tokenizer_512.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-punctuation-1024':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_punctuation_tokenizer_1024.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-punctuation-2048':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_punctuation_tokenizer_2048.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-punctuation-4096':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_punctuation_tokenizer_4096.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-punctuation-8192':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_punctuation_tokenizer_8192.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-punctuation-16384':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_punctuation_tokenizer_16384.json"
        )
    elif config.tokenizer_type == 'bpe-metaspace-punctuation-32768':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_metaspace_punctuation_tokenizer_32768.json"
        )
    elif config.tokenizer_type == 'bpe-whitespace-punctuation-256':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_whitespace_punctuation_tokenizer_256.json"
        )
    elif config.tokenizer_type == 'bpe-whitespace-punctuation-512':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_whitespace_punctuation_tokenizer_512.json"
        )
    elif config.tokenizer_type == 'bpe-whitespace-punctuation-1024':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_whitespace_punctuation_tokenizer_1024.json"
        )
    elif config.tokenizer_type == 'bpe-whitespace-punctuation-2048':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_whitespace_punctuation_tokenizer_2048.json"
        )
    elif config.tokenizer_type == 'bpe-whitespace-punctuation-4096':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_whitespace_punctuation_tokenizer_4096.json"
        )
    elif config.tokenizer_type == 'bpe-whitespace-punctuation-8192':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_whitespace_punctuation_tokenizer_8192.json"
        )
    elif config.tokenizer_type == 'bpe-whitespace-punctuation-16384':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_whitespace_punctuation_tokenizer_16384.json"
        )
    elif config.tokenizer_type == 'bpe-whitespace-punctuation-32768':
        tokenizer = Tokenizer.from_file(
            "custom_tokenizers/bpe_whitespace_punctuation_tokenizer_32768.json"
        )
    else:
        raise ValueError("Invalid tokenizer - Choose one from the command line arguments")

    return tokenizer


def get_model(config: Config):
    if (
        config.architecture is not None
        and config.architecture.lower() == "decoder-transformer"
    ):
        model = DecoderTransformer(config=config).to(config.device)
    elif config.architecture is not None and config.architecture.lower() == "rnn":
        model = RNNBaseline(config=config).to(config.device)
    else:
        raise ValueError("Invalid architecture")

    if config.pretrained_model:
        model.load_state_dict(
            torch.load(config.pretrained_model, map_location=config.device)
        )

    return model


def get_optimizer(model, config: Config):
    if config.optimizer_type is not None and config.optimizer_type.lower() == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type is not None and config.optimizer_type.lower() == 'rmsprop':
        optimizer = RMSprop(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer_type is None or config.optimizer_type.lower() == 'adamw':
        optimizer = configure_adamw_optimizer(
            model=model,
            weight_decay=config.weight_decay,
            learning_rate=config.lr,
            betas=(config.beta1, config.beta2),
            device_type=config.device,
        )
    else:
        raise ValueError("Invalid optimizer")

    return optimizer


def get_scheduler(optimizer, config: Config):
    if config.scheduler_type is not None and config.scheduler_type.lower() == 'noam':
        scheduler = NoamLR(
            optimizer=optimizer,
            dim_model=config.d_model,
            warmup_steps=config.warmup_steps,
        )
    elif config.scheduler_type is not None and config.scheduler_type.lower() == 'cosine':
        scheduler = CosineRateScheduler(
            optimizer=optimizer,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
            lr_decay_iters=config.lr_decay_iters,
            learning_rate=config.lr,
        )
    elif (
        config.scheduler_type is not None and config.scheduler_type.lower() == 'constant'
    ):
        scheduler = ConstantScheduler(
            optimizer=optimizer, warmup_steps=config.warmup_steps, learning_rate=config.lr
        )
    elif (
        config.scheduler_type is None
        or config.scheduler_type.lower() == 'none'
        or config.scheduler_type == ''
    ):
        scheduler = None
    else:
        raise ValueError("Invalid scheduler")

    return scheduler


def get_dataset(tokenizer, config: Config):
    if config.dataset is not None and config.dataset.lower() == 'southpark':
        return SouthParkScriptsDataset(
            tokenizer, config, folder_path='southpark_scripts/all_scripts'
        )
    elif config.dataset is not None and config.dataset.lower() == 'cleaned-southpark':
        return SouthParkScriptsDataset(
            tokenizer, config, folder_path='southpark_scripts/all_scripts_cleaned'
        )
    elif config.dataset is not None and config.dataset.lower() == 'gutenberg':
        return GutenbergDataset(tokenizer, config)
    elif config.dataset is not None and config.dataset.lower() == 'shuffled-gutenberg':
        return ShuffledGutenbergDataset(tokenizer, config)
    else:
        raise ValueError("Invalid dataset")


def get_dataloader(
    training_data_set: Dataset,
    validation_data_set: Dataset,
    test_data_set: Dataset,
    config: Config,
    NUM_WORKERS: int = 0,
):
    training_data_loader = DataLoader(
        training_data_set,
        batch_size=config.batch_size,
        shuffle=False  # shuffled-gutenberg is already shuffled, save memory
        if config.dataset == 'shuffled-gutenberg'
        else True,
        num_workers=NUM_WORKERS,
    )

    validation_data_loader = DataLoader(
        validation_data_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )  # no need to shuffle validation data

    test_data_loader = DataLoader(
        test_data_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )  # no need to shuffle test data

    return training_data_loader, validation_data_loader, test_data_loader


def get_split_ratios(config: Config):
    split_ratios_map = {
        '80-10-10': [0.8, 0.1, 0.1],
        '90-5-5': [0.9, 0.05, 0.05],
        '95-2.5-2.5': [0.95, 0.025, 0.025],
        '99-0.5-0.5': [0.99, 0.005, 0.005],
        '99.9-0.05-0.05': [0.999, 0.0005, 0.0005],
    }

    if config.split_ratios in split_ratios_map:
        return split_ratios_map[config.split_ratios]
    else:
        raise ValueError(f"Data split {config.split_ratios} is not supported.")


def save_model_params(model, save_path, wandb_obj, config: Config):
    torch.save(model.state_dict(), save_path)

    # save model to wandb
    if config.log_wandb:
        os.makedirs(os.path.join(wandb_obj.run.dir + "/models_params/"), exist_ok=True)
        # copy to wandb files
        shutil.copy(
            src=save_path, dst=os.path.join(wandb_obj.run.dir + "/models_params/")
        )
