from .base_runner import BaseRunner
from .. import utils
from ..ardm import graph_ardm
from ..discriminator import discriminator


class SamplingRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def init_model(self, config):
        state_dict = utils.get_pretrained_checkpoint(config["load"])
        old_config = state_dict['config']
        old_config['device'] = config['device']
        self.config["task"] = old_config["task"]
        self.config["graph_kernel"] = old_config["graph_kernel"]
        self.old_config = old_config

        gen_model = graph_ardm.GraphARDM(old_config)
        gen_model.load_state_dict(state_dict['state_dict'])
        if self.config["load_discriminator"] is not None:
            discriminator_checkpoint = utils.get_pretrained_checkpoint(config["load_discriminator"])
            discriminator_config = discriminator_checkpoint["config"]
            assert discriminator_config["task"] == old_config["task"], "Loaded models trained on different tasks"
            discriminator_config["device"] = config["device"]
            if "discriminator" not in discriminator_config:
                discriminator_config["discriminator"] = "transformer"  # backward compatibility
            discriminator_model = discriminator.Discriminator(discriminator_config)
            discriminator_model.load_state_dict(discriminator_checkpoint["state_dict"])
            discriminator_model = discriminator_model.to(config["device"])
            discriminator_guidance = discriminator.guidance_dict[config["guidance_mode"]](gen_model,
                                                                                          discriminator_model,
                                                                                          config)
            self.model = discriminator_guidance
        else:
            self.model = gen_model

    def init_optimizer(self, config):
        return


class EvaluationRunner(SamplingRunner):
    def run(self):
        utils.evaluate_molecule_samples(self.model, self.config)


class GenerationRunner(SamplingRunner):
    def run(self):
        utils.create_fake_molecule_dataset(self.model, self.old_config)





