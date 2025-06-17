from typing import Tuple
from dataclasses import dataclass

@dataclass
class TrainingArgs:
    """
    learning_rate (float): Hyperparameter which controls how big an optimizer step is.
    epochs (int): Number of iterations through the training loop.
    batch_size (int): Number of examples being processed in parallel.
    epsilon (float): Small epsilon value to ensure numerical stability in Adam optimizer.
    clip_grad_norm (float): Value to clip gradient's L2 Norms. Combats exploding gradients problem.
    weight_decay (float): Regularization technique to ensure weight updates are not too large.
    betas (float): Exponential weighted moving averages used in Adam optimizers.
    warmup_epochs (int): Number of warmup epoch if using a learning rate scheduler.
    eta_min (float): Minimum learning rate using in the CosineAnnealingLR scheduler.
    num_workers (int): Number of processes to load data in parallel.
    pin_memory (bool): Speeds up data transfer to the GPU by using pinned (page-locked) memory.
    persistent_workers (bool): Whether to keep workers alive from epoch to epoch.
    grad_accum_steps (int): Number of gradient accumulation steps to stimulate a larger batch size.
    logging_steps (int): Number of training steps before each log.
    eval_steps (int): Model is tested on a validation set every time this value is reached.
    save_steps (int): Save model checkpoint every time this value is reached.
    """
    learning_rate: float = 2e-4
    epochs: int = 300
    batch_size: int = 256
    epsilon: float = 1e-6
    clip_grad_norm: float = 1.0
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_epochs: int = 50
    eta_min: float = 6e-7
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    grad_accum_steps: int = 4
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
