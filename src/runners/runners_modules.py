import torch
import torch.nn as nn

from ..logging import loggers

from . import ardm_trainer, discriminator_trainer, sampling_runner


optimizers = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

loss_fns = {
    "mse": nn.MSELoss
}

loggers_dict = {
    "wandb": loggers.WandBLogger,
    "none": loggers.NoOpLogger
}

runners_dict = {
    "train_ardm": ardm_trainer.ARDMTrainer,
    "train_discriminator": discriminator_trainer.DiscriminatorTrainer,
    "evaluate": sampling_runner.EvaluationRunner,
    "generate": sampling_runner.GenerationRunner
}