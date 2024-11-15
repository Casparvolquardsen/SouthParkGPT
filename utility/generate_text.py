import torch


# Inference Function for generating text, single greedy search (one sequence)
@torch.no_grad()
def generate(
    model,
    tokens,
    seq_len,
    tokenizer,
    max_len_generate=100,
    temperature=1,
    device='cpu',
    print_incremental_output=False,
) -> str:
    model.eval()
    tokens = tokens.unsqueeze(0).to(device)
    new_tokens = []
    current_buffer = []
    for _ in range(max_len_generate):
        trimmed_tokens = tokens[
            :, -seq_len:
        ]  # take the last token but hold the sequence length (B, T)
        logits = model(trimmed_tokens)

        logits = logits[:, -1, :]  # take last token. from shape (B, T, C) to (B, C)

        # Apply temperature scaling to the logits
        logits = logits / temperature

        probs = torch.nn.functional.softmax(logits, dim=-1)  # shape (B, C)
        next_token = torch.multinomial(probs, num_samples=1)  # shape (B, 1)

        # append next token ix to the solution sequence so far
        tokens = torch.cat([tokens, next_token], dim=-1)  # shape (B, T+1)
        new_tokens.append(next_token.item())

        # sample in real time
        if print_incremental_output:
            next_token_id = next_token.item()
            if tokenizer.decode([next_token_id]) == '\n':
                current_buffer += [next_token_id]

                print(tokenizer.decode(current_buffer), end='', flush=True)

                current_buffer = []
            else:
                current_buffer += [next_token_id]

    model.train()
    return tokenizer.decode(new_tokens)  # decode the token to text


@torch.no_grad()
def beam_search_generate(
    model,
    tokens,
    seq_len,
    tokenizer,
    max_len_generate=100,
    temperature=1,
    beam_width=3,
    device='cpu',
) -> str:
    model.eval()
    tokens = tokens.unsqueeze(0).to(device)

    # Initialize the beams
    beams = [(tokens, 0.0)]  # Each beam is a tuple (tokens, cumulative_log_prob)

    for _ in range(max_len_generate):
        all_candidates = []

        for beam_tokens, beam_log_prob in beams:
            input = beam_tokens[:, -seq_len:]  # take the last tokens with sequence length
            logits = model(input)
            logits = logits[:, -1, :]  # take last token. from shape (B, T, C) to (B, C)

            # Apply temperature scaling to the logits
            logits = logits / temperature

            probs = torch.nn.functional.softmax(logits, dim=-1)  # shape (B, C)
            top_probs, top_indices = torch.topk(
                probs, beam_width, dim=-1
            )  # shape (B, beam_width)

            for i in range(beam_width):
                next_token = top_indices[:, i : i + 1]  # shape (B, 1)
                next_log_prob = torch.log(top_probs[:, i : i + 1])  # shape (B, 1)

                new_tokens = torch.cat(
                    [beam_tokens, next_token], dim=-1
                )  # shape (B, T+1)
                new_log_prob = beam_log_prob + next_log_prob.item()

                all_candidates.append((new_tokens, new_log_prob))

        # Sort candidates by log probability and select top `beam_width` candidates
        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

    model.train()
    best_beam = beams[0][0]
    return tokenizer.decode(best_beam.squeeze().tolist())  # decode the token to text
