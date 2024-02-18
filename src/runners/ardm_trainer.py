import torch
import numpy as np
import os
import torch.nn as nn

from .base_runner import BaseRunner
from . import runners_modules
from ..ardm import graph_ardm

from .. import utils


class LossFunction(nn.Module):
    def forward(self, out):
        return out["loss"]


class ARDMTrainer(BaseRunner):
    def __init__(self, config):
        super().__init__(config)

        if config["load"]:
            self.load_checkpoint(config)
        else:
            self.epoch = None
        self.init_epoch_step()

    def init_epoch_step(self):
        self.num_steps_per_epoch = len(self.train_loader)
        if self.epoch is not None:
            # epoch is from saved model
            # lag between saved epoch number and number of epochs that have been performed

            self.step = self.epoch * self.num_steps_per_epoch + self.num_steps_per_epoch
            self.epoch = self.epoch + 1
        else:
            self.step = 0
            self.epoch = 0

    def load_checkpoint(self, config):
        checkpoint_dict = utils.get_pretrained_checkpoint(config["load"], best=False)
        self.model.load_state_dict(checkpoint_dict["state_dict"])
        self.epoch = checkpoint_dict["epoch"]

    def init_model(self, config):
        self.model = graph_ardm.GraphARDM(config)

    def init_optimizer(self, config):
        self.optimizer = runners_modules.optimizers[config["optimizer"]](self.model.parameters(),
                                                                          lr=config["lr"],
                                                                          weight_decay=config["l2_reg"])

    def train(self):
        print("Starting training of ARDM")
        # Loss function
        criterion = LossFunction().to(self.device)

        # Set model to train mode
        self.model.train(True)
        best_val_metric = None

        for epoch_i in range(self.epoch, self.num_epochs):
            epoch_losses = []

            for batch_i, batch in enumerate(self.train_loader):
                # Train
                self.step += 1
                self.optimizer.zero_grad()
                batch = batch.to(self.device)

                out = self.model(batch)
                loss = criterion(out)

                loss.backward()
                self.optimizer.step()

                # Save and log loss
                epoch_losses.append(loss.item())

                self.logger.log({
                                    "train/step_loss": loss.item(),
                                } | out.get('logging_dict', {}), step=self.step)

                # store current state
                current_state = {
                    'epoch': epoch_i,
                    'state_dict': self.model.state_dict(),
                    'config': self.config
                }

                # Evaluate
                if ((epoch_i * self.num_steps_per_epoch + batch_i + 1) % self.config["val_interval"]) == 0:
                    self.model.eval()
                    _ = utils.evaluate_molecule_samples(self.model, self.config, val=True, logger=self.logger,
                                                        step=self.step)
                    self.model.train()

            mean_epoch_loss = np.mean(epoch_losses)

            self.logger.log({
                "train/epoch_loss": mean_epoch_loss,
            }, step=self.step)

            if ((epoch_i + 1) % self.config["print_interval"]) == 0:
                print(f"Epoch {epoch_i}\tloss: {mean_epoch_loss:.4}")
            self.logger.save_cehckpoint(current_state, best=False)
            self.logger.log({"epoch": epoch_i}, step=self.step)

        print('Training finished!')
        self.model.train(False)

        # Update summary with values corresponding to saved best model
        self.logger.update_summary({"best_val_metric": best_val_metric})

    def run(self):
        self.train()
