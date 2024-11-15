import torch
from torch import Tensor
from torcheval.metrics.functional.text import bleu_score

from models import generate_look_ahead_mask


def evaluate_loss_acc(model, data_loader, criterion, config):
    with torch.no_grad():
        model.eval()
        num_batches = len(data_loader)

        accumulated_loss = 0
        accuracy = 0
        top_10_accuracy = 0

        for batch_idx, (src_batch, tgt_batch) in enumerate(data_loader):
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

            accumulated_loss += loss.item()

            # Calculare top k accuracys
            accuracy += evaluate_top_k_accuracy(output_batch, tgt_batch, 1)
            top_10_accuracy += evaluate_top_k_accuracy(output_batch, tgt_batch, 10)

        return (
            accumulated_loss / num_batches,
            accuracy / num_batches,
            top_10_accuracy / num_batches,
        )


# logits: [batch_size*context_len, num_classes], target: [batch_size*context_len], config: Config, k: int
def evaluate_top_k_accuracy(logits: Tensor, target: Tensor, k: int) -> float:
    _, tk = torch.topk(logits, k, dim=1)

    correct_tokens = torch.eq(tk, target.view(-1, 1)).any(dim=1)

    top_k_acc = correct_tokens.float().mean().item() * 100

    return top_k_acc


def evaluate_bleu_score(prediction: str, reference: str):
    bleu = bleu_score(prediction, [reference])
    return bleu
