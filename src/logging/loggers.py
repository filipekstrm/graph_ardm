import os
import wandb
import torch

from . import logging_details


class BaseLogger:
    def __init__(self, config):
        print("Setting up logger")
        if not os.path.isdir("checkpoints"):
            os.mkdir("checkpoints")
        if not os.path.isdir("logs"):
            os.mkdir("logs")

    def log(self, data_to_log, step):
        raise NotImplementedError

    def save_checkpoint(self, dict_to_save, best):
        raise NotImplementedError

    def update_summary(self, log):
        raise NotImplementedError


class NoOpLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)

    def log(self, data_to_log, step):
        return

    def save_checkpoint(self, dict_to_save, best):
        return

    def update_summary(self, log):
        return


class WandBLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)

        self.checkpoint_dir = os.path.join("checkpoints", config["run_name"])
        assert not os.path.isdir(self.checkpoint_dir), "Run directory already exists"
        os.mkdir(self.checkpoint_dir)

        self.log_dir = os.path.join("logs", config["run_name"])
        assert not os.path.isdir(self.log_dir), "Log directory already exists"
        os.mkdir(self.log_dir)
        wandb.init(project=logging_details.WANDB_PROJECT,
                   config=config,
                   name=config['run_name'],
                   dir=self.log_dir
                   )

    def log(self, data_to_log, step):
        wandb.log(data_to_log, step=step)

    def save_checkpoint(self, dict_to_save, best):
        if best:
            name = "best_parameters.pt"
        else:
            name = "parameters.pt"
        torch.save(dict_to_save, os.path.join(self.checkpoint_dir, name))

    def update_summary(self, log):
        for key, value in log.items():
            wandb.run.summary[key] = value

