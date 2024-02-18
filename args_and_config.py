import argparse
import os
import json

import torch

from datetime import datetime


def get_parser():
    """
    Function to obtain the config-dictionary with all parameters to be used in training/testing
    :return: config - dictionary with all parameters to be used
    """
    parser = argparse.ArgumentParser(description='Graph Autoregressive Diffusion Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # If config file should be used
    parser.add_argument("--config", type=str, help="Config file to read run config from")

    # General
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random number generator")
    parser.add_argument("--mode", type=str.lower,
                        help="What mode to run in (train_ardm, train_discriminator, evaluate, generate)")
    parser.add_argument("--load", type=str.lower,
                        help="Load parameters from pre-trained model (generator)")
    parser.add_argument("--load_discriminator", type=str.lower,
                        help="Load parameters of discriminator")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Whether to disable cuda, even if available')
    parser.add_argument("--task", type=str.lower,
                        help="What task to perform (qm9, qm9_no_h, moses")
    parser.add_argument("--num_samples", type=int,
                        help="Number of samples to evaluate on")
    parser.add_argument("--logger", type=str.lower,
                        help="Logger type (none or wandb)")

    # Model related
    parser.add_argument("--n_hidden_layers", type=int, default=3,
                        help="Number of hidden layers in MLP")
    parser.add_argument("--activation_function", type=str.lower, default='relu',
                        help="MLP activation function")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Dimension for hidden layers")
    parser.add_argument("--batch_norm", type=int, default=0,
                        help="Use batch norm in MLP")

    # Graph Transformer
    parser.add_argument("--hidden_mlp_dim_x", type=int, default=256,
                        help="Hidden mlp dim for nodes")
    parser.add_argument("--hidden_mlp_dim_e", type=int, default=128,
                        help="Hidden mlp dim for edges")
    parser.add_argument("--hidden_mlp_dim_y", type=int, default=1,
                        help="Hidden mlp dim for y. Not used, and hence should be 1 for speed")

    parser.add_argument("--hidden_dim_dx", type=int, default=256,
                        help="Hidden dim for dx")
    parser.add_argument("--hidden_dim_de", type=int, default=64,
                        help="Hidden mlp dim for de")
    parser.add_argument("--hidden_dim_dy", type=int, default=1,
                        help="Hidden mlp dim for dy. Not used, and hence should be 1 for speed")

    parser.add_argument("--dim_ffX", type=int, default=256,
                        help="Feedforward dim, nodes")
    parser.add_argument("--dim_ffE", type=int, default=128,
                        help="Feedforward dim, edges")
    parser.add_argument("--dim_ffy", type=int, default=1,
                        help="Feedforward dim, y. Not used, and hence should be 1 for speed")

    parser.add_argument("--n_head", type=int, default=8,
                        help="Number of Transformer heads")

    parser.add_argument("--node_lambda", type=float,
                        help="Node lambda", default=1.0)

    parser.add_argument("--edge_lambda", type=float,
                        help="Edge lambda", default=1.0)

    parser.add_argument("--node_next_odds", type=float,
                        help="For NEsN, uppweighing the probability of sampling a node next (or, mask 0 edges)",
                        default=1.0)

    parser.add_argument("--graph_kernel", type=str.lower, choices=['autoreg', 'nen', 'ne', 'partialautoreg'],
                        help="Which graph kernel to use", default='autoreg')

    # Data related
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train set ratio, if dataset is not already split")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation set ratio, if dataset is not already split")

    parser.add_argument("--filter", type=int, default=1,
                        help="If filtering dataset")

    # Training
    parser.add_argument("--epochs", type=int,
                        help="How many epochs to train for", default=100)
    parser.add_argument("--val_interval", type=int,
                        help="How often to evalute on validation set (number of gradient steps)", default=10000)
    parser.add_argument("--print_interval", type=int,
                        help="How often to print (number of epochs)", default=1)
    parser.add_argument("--l2_reg", type=float, default=0,
                        help="L2-regularization added to cost function (aka weight decay)")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for training", default=32)
    parser.add_argument("--lr", type=float,
                        help="Learning rate", default=1e-3)
    parser.add_argument("--optimizer", type=str.lower,
                        help="Optimizer to use for training", default="adam")
    parser.add_argument("--scheduler", type=int,
                        help="If to use Cosine annealing")
    parser.add_argument("--lr_min", type=float, default=0,
                        help="eta_min if using cosine scheduler")

    # Discriminator
    parser.add_argument("--discriminator", type=str.lower, default="transformer",
                        help="Which architecture for the discriminator")

    parser.add_argument("--guidance_mode", type=str.lower,
                        help="Which mode to run the discriminator guidance (ardg, fadg, bsdg)")

    parser.add_argument("--max_masked_percentage", type=float, default=1.0,
                        help="Maximum percentage of elements that should be masked")

    parser.add_argument("--gradual_max_masked_percentage", type=int, default=0,
                        help="Have a gradual increase in max percentage of masked elements")
    parser.add_argument("--increase_masked_percentage_every", type=int, default=7500,
                        help="Number of steps between increasing masked percentage")

    parser.add_argument("--pool_only_unmasked", type=float, default=0,
                        help="Pool only unmasked representations")
    parser.add_argument("--discriminator_tempering", type=int, default=0,
                        help="Temper discriminator")

    parser.add_argument("--discriminator_temp_constant", type=float, default=1.0,
                        help="Discriminator temp constant if using tempering")

    parser.add_argument("--encode_time", type=int, default=0,
                        help="Encode timestep")

    parser.add_argument("--gen_data_path", type=str.lower,
                        help="Path to generated data")

    # Particle filter
    parser.add_argument("--pf_resampling", type=str.lower, default="systematic",
                        help="type of resampling in particle filter")

    parser.add_argument("--ess_ratio", type=float, default=0.7,
                        help="ESS ratio for resampling (i.e, the number N_thresh/N")
    parser.add_argument("--debug_pf", type=int, default=0,
                        help="Run PF in debug mode")
    parser.add_argument("--same_order", type=int, default=1,
                        help="Same order for all particles")

    parser.add_argument("--num_particles", type=int, default=-1,
                        help="Number of particles per sample. -1 if using a full batch with particles")

    return parser


def get_included_config(file_path):
    with open(file_path, "r") as json_file:
        config = json.load(json_file)
        if "include" in config:
            included_config = get_included_config(config["include"])
            assert "include" not in included_config
            included_config.update(config)
            del included_config["include"]
            config = included_config.copy()
    return config


def get_config():
    # First parse, and include default values
    default_parser = get_parser()
    default_args = default_parser.parse_args()
    config = vars(default_args)

    # Now update with arguments from config file
    if default_args.config:
        assert os.path.exists(default_args.config), f"No config file: {default_args}"
        with open(default_args.config) as json_file:
            config_from_file = json.load(json_file)
        if "include" in config_from_file:  # ability to include another json to avoid having to specify multiple things
            included_config = get_included_config(config_from_file["include"])
            included_config.update(config_from_file)
            config_from_file = included_config.copy()
            del config_from_file["include"]
        unknown_options = set(config_from_file.keys()).difference(set(config.keys()))
        unknown_error = "\n".join(["Unknown option in config file: {}".format(opt)
                                   for opt in unknown_options])
        assert (not unknown_options), unknown_error
        config.update(config_from_file)

    # Now parse again, but without any default values
    command_line_parser = argparse.ArgumentParser(parents=[default_parser], add_help=False)
    command_line_parser.set_defaults(**{key: None for key in config.keys()})
    command_line_args = command_line_parser.parse_args()

    # now overwrite values in config with those from command line
    flags_from_command_line = []
    config_from_command_line = vars(command_line_args)
    for key, value in config_from_command_line.items():
        if value is not None:
            if key == "config":
                value = str(value)
                value = os.path.splitext(value)[0]
                value = "-".join(os.path.normpath(value).split(os.path.sep))
            if key != "logger":
                flags_from_command_line.append(f'{key}={value}' if key != 'config' else str(value))
            config[key] = value

    config['cuda'] = not config['disable_cuda'] and torch.cuda.is_available()
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    config['run_name'] = '_'.join([dt_string] + flags_from_command_line)
    return config


if __name__ == '__main__':
    print(get_config())
