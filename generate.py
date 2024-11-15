import logging
import os
import time

import torch

from utility import config_from_args, generate, get_args
from utility.utils import get_model, get_tokenizer, model_size, set_seed

LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = "models_params"

if __name__ == "__main__":
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

    ######## MODEL HYPERPARAMS ########
    tokenizer = get_tokenizer(config)
    config.vocab_size = tokenizer.get_vocab_size()

    # create model
    model = get_model(config)

    model_size_bits = model_size(model)
    logger.info(f"Model size: {model_size_bits} bit | {model_size_bits / 8e6:.2f} MB")

    if args.prompt_file:
        with open(args.prompt_file, 'r') as file:
            prompt_str = file.read()
    else:
        prompt_str = args.prompt

    prompt = torch.tensor(tokenizer.encode(prompt_str).ids)

    start_time = time.time()
    print("Prompt: ", prompt_str)
    print("Generated: ", end='')
    generated_text = generate(
        model=model,
        tokens=prompt,
        seq_len=config.context_len,
        tokenizer=tokenizer,
        max_len_generate=config.max_len_generate,
        temperature=config.temperature,
        device=config.device,
        print_incremental_output=True,
    )
    end_time = time.time()
    logger.info(f"\nTime taken to generate: {(end_time - start_time) / 60} Minutes")
