import torch
import torch.nn as nn
import numpy as np

from .base_runner import BaseRunner
from . import runners_modules

import itertools

from .. import utils

from ..dataset import dataset

from ..discriminator import discriminator


class DiscriminatorTrainer(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.init_scheduler()

    def init_model(self, config):
        self.model = discriminator.Discriminator(config)
        if config["load"]:  # initialize discriminator from generator
            checkpoint_dict = utils.get_pretrained_checkpoint(config["load"], best=True)
            utils.load_discriminator_checkpoint(self.model, checkpoint_dict["state_dict"])
            print("Loaded initialization from pre-trained generator")

    def init_optimizer(self, config):
        self.optimizer = runners_modules.optimizers[config["optimizer"]](self.model.parameters(),
                                                                         lr=config["lr"],
                                                                         weight_decay=config["l2_reg"])

    def init_scheduler(self):
        self.scheduler = None  # For now

    def init_dataloaders(self, config):
        (self.train_loader_real, self.val_loader_real, self.test_loader_real, self.train_loader_gen, self.val_loader_gen,
         self.test_loader_gen) = dataset.get_dataloaders(config)

    def train(self):
        print("Starting training of discriminator")
        # Loss function
        criterion = nn.BCEWithLogitsLoss().to(self.device)

        # Set model to train mode
        self.model.train(True)
        best_val_metric = None

        self.step = 0
        num_steps_per_epoch = len(self.train_loader_real)
        train_loader_gen_iter = itertools.cycle(
            self.train_loader_gen)  # to handle the case when the generated set is smaller

        for epoch_i in range(self.num_epochs):
            epoch_losses = []
            for batch_i, batch_real in enumerate(self.train_loader_real):
                batch_gen = next(train_loader_gen_iter)
                # Train
                self.step += 1
                self.optimizer.zero_grad()
                batch_real = batch_real.to(self.device)
                batch_gen = batch_gen.to(self.device)

                out_real = self.model(batch_real)
                loss_real = criterion(out_real, torch.ones_like(out_real))

                out_gen = self.model(batch_gen)
                loss_gen = criterion(out_gen, torch.zeros_like(out_gen))

                bs_real = out_real.shape[0]
                bs_gen = out_gen.shape[0]

                loss = (loss_gen * bs_gen + loss_real * bs_real) / (bs_real + bs_gen)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.logger.log({"train/lr": self.scheduler.get_last_lr()[0]}, step=self.step)
                    self.scheduler.step()

                # Save and log loss
                epoch_losses.append(loss.item())

                self.logger.log({
                    "train/step_loss": loss.item(), "train/step_loss_gen": loss_gen.item(),
                    "train/step_loss_real": loss_real.item()
                }, step=self.step)

                if hasattr(self.model.kernel, "step_update"):
                    self.logger.log({"max_masked_percentage": self.model.kernel.get_max_percentage()}, step=self.step)
                    self.model.kernel.step_update(self.step)
                # store current state
                current_state = {
                    'epoch': epoch_i,
                    'state_dict': self.model.state_dict(),
                    'config': self.config
                }

                # Evaluate
                if ((epoch_i * num_steps_per_epoch + batch_i + 1) % self.config["val_interval"]) == 0:
                    self.model.eval()
                    val_loss, val_loss_real, val_loss_gen = utils.evaluate_discriminator(self.model, self.config,
                                                                                         self.val_loader_real,
                                                                                         self.val_loader_gen)
                    self.logger.log({
                        "val/bce": val_loss.item(), "val/bce_gen": val_loss_gen.item(),
                        "val/bce_real": val_loss_real.item()
                    }, step=self.step)

                    if (best_val_metric is None) or (val_loss < best_val_metric):
                        # model is improved, save it as best
                        best_val_metric = val_loss
                    self.model.train()

                # No reshuffling when using itertools.cycle on a dataloader with shuffle, so it is done here
                if (self.step % len(self.train_loader_gen)) == 0:
                    train_loader_gen_iter = itertools.cycle(self.train_loader_gen)

            mean_epoch_loss = np.mean(epoch_losses)

            self.logger.log({
                "train/epoch_loss": mean_epoch_loss,
            }, step=self.step)

            if ((epoch_i + 1) % self.config["print_interval"]) == 0:
                print(f"Epoch {epoch_i}\tloss: {mean_epoch_loss:.4}")
            # Save this epoch's parameters
            self.logger.save_checkpoint(current_state, best=False)
            self.logger.log({"epoch": epoch_i}, step=self.step)

        print('Training finished!')
        self.model.train(False)

        # Update summary with values corresponding to saved best model
        self.logger.update_summary({"best_val_metric": best_val_metric})

    def run(self):
        self.train()
