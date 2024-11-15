import math

from torch.optim.lr_scheduler import LRScheduler


class NoamLR(LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """

    def __init__(self, optimizer, dim_model, warmup_steps):
        self.warmup_steps = warmup_steps
        self.dim_model = dim_model
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)

        scale = self.dim_model**0.5 * min(
            last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]


# from nanoGPT
class CosineRateScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, lr_decay_iters, learning_rate, min_lr):
        self.warmup_steps = warmup_steps
        self.lr_decay_iters = lr_decay_iters
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        # 1) Linear warmup for warmup_iters steps
        if self.last_epoch < self.warmup_steps:
            return [
                self.learning_rate * self.last_epoch / self.warmup_steps
                for _ in self.optimizer.param_groups
            ]

        # 2) If iteration > lr_decay_iters, return minimum learning rate
        if self.last_epoch > self.lr_decay_iters:
            return [self.min_lr for _ in self.optimizer.param_groups]

        # 3) In between, use cosine decay down to minimum learning rate
        decay_ratio = (self.last_epoch - self.warmup_steps) / (
            (self.lr_decay_iters - self.warmup_steps) + 1e-8
        )  # Avoid division by zero w/ 1e-8
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Coeff ranges from 0 to 1
        return [
            self.min_lr + coeff * (self.learning_rate - self.min_lr)
            for _ in self.optimizer.param_groups
        ]


class ConstantScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, learning_rate):
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.learning_rate * self.last_epoch / self.warmup_steps
                for _ in self.optimizer.param_groups
            ]
        return [self.learning_rate for _ in self.optimizer.param_groups]
