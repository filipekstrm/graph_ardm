import torch
import os

import torch_geometric as ptg

from . import moses_dataset as moses
from .generated_dataset import GeneratedDataset
from . import qm9_dataset


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path


def get_target_dataset_size(config):
    task = config["task"]
    if task in ["qm9", "qm9_no_h"]:
        return 130831
    elif task == "moses":
        return 1936962
    else:
        raise ValueError(f"Dataset size for {task} is unknown")


def get_dataloaders(config):
    # Load train, val, test dataset
    if config["task"] == "moses":
        train_dataset, val_dataset, test_dataset = moses.get_datasets(config)
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        test_size = len(test_dataset)
    elif config["task"].startswith("qm9"):
        dataset = qm9_dataset.QM9Dataset(config)
        num_samples = len(dataset)
        train_size = int(config['train_ratio']*num_samples)
        val_size = int(config['val_ratio']*num_samples)
        test_size = num_samples - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                 [train_size, val_size, test_size],
                                                                                 generator=torch.Generator().manual_seed(config['seed']))
    else:
        raise ValueError(f"Unknown task: {config['task']}")

    train_loader = ptg.loader.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = ptg.loader.DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = ptg.loader.DataLoader(test_dataset, batch_size=config['batch_size'])

    # additionally, if training discriminator, create datasets and dataloaders with fake data
    if config['mode'] == 'train_discriminator':
        gen_dataset = GeneratedDataset(config)

        num_gen_samples = len(gen_dataset)
        smaller_gen_dataset = num_gen_samples != (train_size + val_size + test_size)
        if smaller_gen_dataset:
            train_size_gen = int(config['train_ratio'] * num_gen_samples)
            val_size_gen = int(config['val_ratio'] * num_gen_samples)
            test_size_gen = num_gen_samples - train_size_gen - val_size_gen
        else:
            train_size_gen, val_size_gen, test_size_gen = train_size, val_size, test_size

        train_dataset_gen, val_dataset_gen, test_dataset_gen = torch.utils.data.random_split(gen_dataset,
                                                                                             [train_size_gen, val_size_gen, test_size_gen],
                                                                                             generator=torch.Generator().manual_seed(
                                                                                                 config['seed']))

        train_loader_gen = ptg.loader.DataLoader(train_dataset_gen, batch_size=config['batch_size'], shuffle=True)
        val_loader_gen = ptg.loader.DataLoader(val_dataset_gen, batch_size=config['batch_size'])
        test_loader_gen = ptg.loader.DataLoader(test_dataset_gen, batch_size=config['batch_size'])
        return train_loader, val_loader, test_loader, train_loader_gen, val_loader_gen, test_loader_gen
    return train_loader, val_loader, test_loader


