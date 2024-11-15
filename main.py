import datetime
import logging
import os
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torchinfo import summary

import wandb
from evaluate import evaluate_loss_acc
from train import train
from utility import config_from_args, config_from_wandb, get_args
from utility.utils import (
    get_dataloader,
    get_dataset,
    get_model,
    get_optimizer,
    get_scheduler,
    get_split_ratios,
    get_tokenizer,
    model_size,
    set_seed,
    split,
)

LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = "models_params"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

NUM_WORKERS = os.getenv('NUM_WORKERS', 8)

if __name__ == '__main__':
    # get arguments
    args = get_args()
    config = config_from_args(args)

    # set seed for reproducibility
    set_seed(args.seed)

    use_cuda = (
        torch.cuda.is_available()
        and torch.backends.cudnn.enabled
        and (args.device == 'cuda' or args.device == 'accelerated')
    )
    use_mps = (
        torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
        and (args.device == 'mps' or args.device == 'accelerated')
    )

    config.device = 'mps' if use_mps else 'cuda' if use_cuda else 'cpu'

    # check whether Metal Performance Shaders (MPS) is available
    logger.info(f"Is MPS built: {torch.backends.mps.is_built()}")
    logger.info(f"Is MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Is CUDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"Model Runs on: {config.device}")

    tokenizer = get_tokenizer(config)

    config.vocab_size = tokenizer.get_vocab_size()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="SouthparkGPT",
        name=f"{datetime.date.today()}_{config.dataset}_{config.architecture}_model",
        # track hyperparameters and run metadata
        config=asdict(config),
        mode="online" if config.log_wandb else "offline",
    )

    # update config with wandb config because it can be overwritten in sweeps
    config = config_from_wandb(wandb.config)
    config.device = 'mps' if use_mps else 'cuda' if use_cuda else 'cpu'
    logger.info(f"Used config: {config}")

    start_date = datetime.date.today()
    save_dir = os.path.join(
        MODEL_SAVE_DIR, f"{start_date}_{wandb.run.id}_{config.architecture}"
    )
    os.makedirs(save_dir, exist_ok=True)

    model = get_model(config)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    model_size_bits = model_size(model)
    logger.info(f"Model size: {model_size_bits} bit | {model_size_bits / 8e6:.2f} MB")

    data_set = get_dataset(tokenizer, config)
    logger.info(f"Number of samples in dataset: {len(data_set)}")

    training_data_set, validation_data_set, test_data_set = split(
        data_set, get_split_ratios(config)
    )

    training_data_loader, validation_data_loader, test_data_loader = get_dataloader(
        training_data_set=training_data_set,
        validation_data_set=validation_data_set,
        test_data_set=test_data_set,
        config=config,
        NUM_WORKERS=NUM_WORKERS,
    )

    logger.info(f"Number of batches in training dataloader: {len(training_data_loader)}")
    logger.info(
        f"Number of batches in validation dataloader: {len(validation_data_loader)}"
    )
    logger.info(f"Number of batches in test dataloader: {len(test_data_loader)}")

    start_training_time = time.time()
    train(
        model,
        training_data_loader=training_data_loader,
        validation_data_loader=validation_data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_data_set=training_data_set,
        validation_data_set=validation_data_set,
        test_data_set=test_data_set,
        tokenizer=tokenizer,
        save_dir=save_dir,
        start_date=start_date,
        config=config,
    )
    end_training_time = time.time()

    wandb.log(
        {"training/total_time (h)": (end_training_time - start_training_time) / 3600}
    )

    # load the best model (based on validation accuracy) and evaluate it on the test set
    best_model_path = os.path.join(
        save_dir,
        f"{start_date}_{wandb.run.id}_best_validation_loss_{config.architecture}.pth",
    )
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=config.device))

    (test_loss, test_accuracy, test_top_10_accuracy) = evaluate_loss_acc(
        model, test_data_loader, nn.CrossEntropyLoss(), config
    )

    logger.info(f"Test loss: {test_loss} - Test accuracy: {test_accuracy}%")
    wandb.log(
        {
            "test/loss": test_loss,
            "test/test_top_10_accuracy": test_top_10_accuracy,
            "test/accuracy": test_accuracy,
        }
    )

    # Print Model Summary
    summary(
        model,
        input_size=(config.batch_size, config.context_len),
        dtypes=[torch.long],
        device=config.device,
    )

    wandb.finish()
