import numpy as np
import torch

from args_and_config import get_config
from src.runners.runners_modules import runners_dict


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config):
    """
    Main function for training or testing
    :param config: dictionary with all parameters to be used for training/testing
    :return:
    """
    # Set all random seeds
    seed_all(config["seed"])

    # Figure out what device to use
    if config['cuda']:
        device = torch.device("cuda")

        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    config["device"] = device

    # initialize runner
    runner = runners_dict[config["mode"]](config)

    # run trainer
    runner.run()


if __name__ == '__main__':
    conf = get_config()
    main(conf)
