import logging
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from evaluate import evaluate_loss_acc, evaluate_top_k_accuracy
from models import generate_look_ahead_mask
from utility import Config
from utility.utils import log_text_evaluation, save_model_params

LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def train(
    model,
    training_data_loader,
    validation_data_loader,
    optimizer,
    scheduler,
    training_data_set,
    validation_data_set,
    test_data_set,
    tokenizer,
    start_date,
    save_dir,
    config: Config,
):
    criterion = nn.CrossEntropyLoss()
    step = 0
    best_val_loss = float("inf")
    training_time = 0
    validation_time = 0

    logger.info('Start training...')
    for epoch in tqdm(range(config.epochs)):
        epoch_loss = 0
        accumulated_loss = 0

        # training loop
        training_start_time = time.time()
        for batch_idx, (src_batch, tgt_batch) in enumerate(training_data_loader):
            model.train()
            step += 1

            src_batch, tgt_batch = (
                src_batch.to(config.device),
                tgt_batch.to(config.device),
            )
            context_len = src_batch.size(1)
            if config.use_attn_mask:
                attn_mask = generate_look_ahead_mask(context_len).to(config.device)
                output_batch = model(src_batch, attn_mask=attn_mask)
            else:
                output_batch = model(src_batch)

            # Flatten the output and target tensors
            output_batch = output_batch.view(-1, output_batch.size(-1))
            tgt_batch = tgt_batch.view(-1)

            # Calculate loss
            loss = criterion(output_batch, tgt_batch)

            # Calculate accuracy

            accuracy = evaluate_top_k_accuracy(output_batch, tgt_batch, 1)
            top_10_accuracy = evaluate_top_k_accuracy(output_batch, tgt_batch, 10)

            # accumulation of loss
            loss /= config.gradient_accumulation_steps
            accumulated_loss += loss.item()
            # Calculate gradients
            loss.backward()
            epoch_loss += loss.item()

            if step % config.gradient_accumulation_steps == 0:
                # gradient clipping
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_value
                    )

                # Update weights
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # Calculate gradient norms
                total_norm = 0
                for p in model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5

                wandb.log(
                    {
                        "training/accumulated_batch_loss": accumulated_loss,
                        "training/gradient_norm": total_norm,
                    },
                    step=step,
                )

                # reset accumulated loss
                accumulated_loss = 0
                optimizer.zero_grad()

            current_lr = optimizer.param_groups[0]['lr']

            wandb.log(
                {
                    "training/batch_loss": loss.item()
                    * config.gradient_accumulation_steps,
                    "training/accuracy": accuracy,
                    "training/top_10_accuracy": top_10_accuracy,
                    "training/lr": current_lr,
                },
                step=step,
            )

            # evaluation every evaluation_interval steps
            if (
                config.evaluation_interval is not None
                and step % config.evaluation_interval == 0
            ):
                training_end_time = time.time()
                training_time += training_end_time - training_start_time
                validation_start_time = time.time()
                (validation_loss, validation_accuracy, validation_top_10_accuracy) = (
                    evaluate_loss_acc(model, validation_data_loader, criterion, config)
                )

                # save model if validation loss is lower than current best
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    save_path = os.path.join(
                        save_dir,
                        f"{start_date}_{wandb.run.id}_best_validation_loss_{config.architecture}.pth",
                    )
                    save_model_params(
                        model=model, save_path=save_path, wandb_obj=wandb, config=config
                    )

                wandb.log(
                    {
                        "validation/loss": validation_loss,
                        "validation/best_loss": best_val_loss,
                        "validation/accuracy:": validation_accuracy,
                    },
                    step=step,
                )

                if config.log_text_samples:
                    num_prompts = 5

                    log_text_evaluation(
                        data_label="training",
                        num_prompts=num_prompts,
                        model=model,
                        tokenizer=tokenizer,
                        dataset=training_data_set,
                        step=step,
                        config=config,
                    )

                    log_text_evaluation(
                        data_label="validation",
                        num_prompts=num_prompts,
                        model=model,
                        tokenizer=tokenizer,
                        dataset=validation_data_set,
                        step=step,
                        config=config,
                    )

                validation_end_time = time.time()
                validation_time += validation_end_time - validation_start_time
                training_start_time = time.time()

        wandb.log({'training/epoch': epoch + 1}, step=step)

        # model parameter snapshot
        if config.save_model_every_epoch:
            save_path = os.path.join(
                save_dir,
                f"{start_date}_{wandb.run.id}_epoch-{epoch + 1}_{config.architecture}.pth",
            )
            save_model_params(
                model=model, save_path=save_path, wandb_obj=wandb, config=config
            )

    wandb.log(
        {
            "training/training_time (h)": training_time / 3600,
            "validation/validation_time (h)": validation_time / 3600,
            "training/training_validation_ratio (training/validation)": f"{training_time / validation_time}/1",
        }
    )
