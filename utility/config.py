from argparse import Namespace
from dataclasses import dataclass


@dataclass
class Config:
    # Architecture parameters
    architecture: str
    bias: bool
    d_model: int
    d_feedforward: int
    dropout: float
    num_heads: int
    num_layers: int
    use_attn_mask: bool
    positional_encoding: str

    # Data parameters
    batch_size: int
    context_len: int
    dataset: str
    num_samples: int
    split_ratios: str
    vocab_size: int

    # Device parameters
    device: str

    # Optimization parameters
    beta1: float
    beta2: float
    gradient_accumulation_steps: int
    gradient_clip_value: float
    lr: float
    lr_decay_iters: int
    min_lr: float
    optimizer_type: str
    scheduler_type: str
    use_gradient_clipping: bool
    warmup_steps: int
    weight_decay: float

    # Training parameters
    epochs: int
    pretrained_model: str
    save_model_every_epoch: bool

    # Tokenization parameters
    tokenizer_type: str

    # Logging parameters
    log_text_samples: bool
    max_len_generate: int
    temperature: float
    beam_search: bool
    log_wandb: bool
    evaluation_interval: int


def config_from_args(args: Namespace) -> Config:
    return Config(
        architecture=args.architecture,
        bias=args.bias,
        d_model=args.d_model,
        d_feedforward=args.d_feedforward,
        dropout=args.dropout,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_attn_mask=args.use_attn_mask,
        positional_encoding=args.positional_encoding,
        batch_size=args.batch_size,
        context_len=args.context_len,
        dataset=args.dataset,
        num_samples=args.num_samples,
        split_ratios=args.split_ratios,
        vocab_size=-1,  # This is set in the training script, after the tokenizer is created
        device=args.device,
        beta1=args.beta1,
        beta2=args.beta2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_value=args.gradient_clip_value,
        lr=float(args.lr),
        lr_decay_iters=args.lr_decay_iters,
        min_lr=float(args.min_lr),
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        use_gradient_clipping=args.use_gradient_clipping,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        pretrained_model=args.pretrained_model,
        save_model_every_epoch=args.save_model_every_epoch,
        tokenizer_type=args.tokenizer,
        log_text_samples=args.log_text_samples,
        max_len_generate=args.max_len_generate,
        temperature=args.temperature,
        beam_search=args.beam_search,
        log_wandb=args.log_wandb,
        evaluation_interval=args.evaluation_interval,
    )


def config_from_wandb(wandb_config) -> Config:
    return Config(
        architecture=wandb_config.architecture,
        bias=wandb_config.bias,
        d_model=wandb_config.d_model,
        d_feedforward=wandb_config.d_feedforward,
        dropout=wandb_config.dropout,
        num_heads=wandb_config.num_heads,
        num_layers=wandb_config.num_layers,
        use_attn_mask=wandb_config.use_attn_mask,
        positional_encoding=wandb_config.positional_encoding,
        batch_size=wandb_config.batch_size,
        context_len=wandb_config.context_len,
        dataset=wandb_config.dataset,
        num_samples=wandb_config.num_samples,
        split_ratios=wandb_config.split_ratios,
        vocab_size=wandb_config.vocab_size,
        device=wandb_config.device,
        beta1=wandb_config.beta1,
        beta2=wandb_config.beta2,
        gradient_accumulation_steps=wandb_config.gradient_accumulation_steps,
        gradient_clip_value=wandb_config.gradient_clip_value,
        lr=wandb_config.lr,
        lr_decay_iters=wandb_config.lr_decay_iters,
        min_lr=wandb_config.min_lr,
        optimizer_type=wandb_config.optimizer_type,
        scheduler_type=wandb_config.scheduler_type,
        use_gradient_clipping=wandb_config.use_gradient_clipping,
        warmup_steps=wandb_config.warmup_steps,
        weight_decay=wandb_config.weight_decay,
        epochs=wandb_config.epochs,
        pretrained_model=wandb_config.pretrained_model,
        save_model_every_epoch=wandb_config.save_model_every_epoch,
        tokenizer_type=wandb_config.tokenizer_type,
        log_text_samples=wandb_config.log_text_samples,
        max_len_generate=wandb_config.max_len_generate,
        temperature=wandb_config.temperature,
        beam_search=wandb_config.beam_search,
        log_wandb=wandb_config.log_wandb,
        evaluation_interval=wandb_config.evaluation_interval,
    )


def config_from_dict(dictionary: dict) -> Config:
    return Config(
        architecture=dictionary["architecture"],
        bias=dictionary["bias"],
        d_model=dictionary["d_model"],
        d_feedforward=dictionary["d_feedforward"],
        dropout=dictionary["dropout"],
        num_heads=dictionary["num_heads"],
        num_layers=dictionary["num_layers"],
        use_attn_mask=dictionary["use_attn_mask"],
        positional_encoding=dictionary["positional_encoding"],
        batch_size=dictionary["batch_size"],
        context_len=dictionary["context_len"],
        dataset=dictionary["dataset"],
        num_samples=dictionary["num_samples"],
        split_ratios=dictionary["split_ratios"],
        vocab_size=dictionary["vocab_size"],
        device=dictionary["device"],
        beta1=dictionary["beta1"],
        beta2=dictionary["beta2"],
        gradient_accumulation_steps=dictionary["gradient_accumulation_steps"],
        gradient_clip_value=dictionary["gradient_clip_value"],
        lr=dictionary["lr"],
        lr_decay_iters=dictionary["lr_decay_iters"],
        min_lr=dictionary["min_lr"],
        optimizer_type=dictionary["optimizer_type"],
        scheduler_type=dictionary["scheduler_type"],
        use_gradient_clipping=dictionary["use_gradient_clipping"],
        warmup_steps=dictionary["warmup_steps"],
        weight_decay=dictionary["weight_decay"],
        epochs=dictionary["epochs"],
        pretrained_model=dictionary["pretrained_model"],
        save_model_every_epoch=dictionary["save_model_every_epoch"],
        tokenizer_type=dictionary["tokenizer_type"],
        log_text_samples=dictionary["log_text_samples"],
        max_len_generate=dictionary["max_len_generate"],
        temperature=dictionary["temperature"],
        beam_search=dictionary["beam_search"],
        log_wandb=dictionary["log_wandb"],
        evaluation_interval=dictionary["evaluation_interval"],
    )
