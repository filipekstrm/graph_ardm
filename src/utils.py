import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

from torch_geometric.data.data import Data

from .dataset import dataset

from .digress_metrics import molecular_metrics as metrics
from . import constants


def get_pretrained_checkpoint(load_path, best=True):
    restored_file = os.path.join(load_path, 'files', constants.BEST_PARAMS_FILE)
    if not os.path.isfile(restored_file) or not best:
        restored_file = os.path.join(load_path, 'files', constants.PARAMS_FILE)
    print("loading, ", restored_file)
    checkpoint = torch.load(restored_file, map_location=lambda storage, loc: storage)
    return checkpoint


def load_generator_into_discriminator(model, state_dict):
    model_state_dict = model.state_dict()
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(f"Skip loading parameter: {k}, "
                      f"required shape: {model_state_dict[k].shape}, "
                      f"loaded shape: {state_dict[k].shape}")
            else:
                model_state_dict[k] = state_dict[k]
        else:
            print(f"Missing parameter {k}")
    model.load_state_dict(model_state_dict)


def load_discriminator_checkpoint(model, state_dict):
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("kernel"):
        state_dict = {"network.graph_encoder" + old_key[12:]: param for old_key, param in state_dict.items()}
        load_generator_into_discriminator(model, state_dict)
    else:
        model.load_state_dict(state_dict)


def generate_molecule_samples(model, num_samples_to_generate):
    model.eval()
    return model.generate(num_samples_to_generate)


def get_evaluation_set_path(config):
    print("Seed used: ", config["seed"])
    path_str = f"{config['task']}_evaluate_{config['graph_kernel']}_{config['guidance_mode']}"
    path_str += f"_seed={config['seed']}"
    if config["num_particles"] > 0:
        path_str += f"_particles:{config['num_particles']}"
    path_str += f"_sampling={config['pf_resampling']}"
    if config["same_order"] != 0:
        path_str += "_sameorder"
    path_str += f"_samples={config['num_samples']}"
    return path_str


def save_molecule_list_to_dir(gen_data, model, path):
    gen_dataset = []
    for graph in gen_data:
        x = F.one_hot(graph[0], num_classes=model.node_output_dim)
        edges = graph[1]
        e_i = []
        e_j = []
        edge_attr = []
        for i in range(edges.shape[0]):
            for j in range(edges.shape[0]):
                if i == j:
                    continue
                if int(edges[i, j]) == 0:
                    continue
                e_i.append(int(i))
                e_j.append(int(j))
                edge_attr.append(int(edges[i, j]))
        e_i = torch.tensor(e_i)
        e_j = torch.tensor(e_j)
        edge_index = torch.vstack([e_j, e_i]).long()
        edge_attr = torch.tensor(edge_attr).long()
        edge_attr = F.one_hot(edge_attr, num_classes=model.edge_output_dim)

        data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        gen_dataset.append(data)
    torch.save(gen_dataset, os.path.join(path, "gen_dataset.pt"))


def get_dataset_size(config):
    task = config["task"]
    if task in ["qm9", "qm9_no_h"]:
        return 130831
    elif task == "moses":
        # return 1936962
        return 200000
    else:
        raise ValueError(f"Dataset size for {task} is unknown")


@torch.no_grad()
def create_fake_molecule_dataset(model, config, size=None, eval=False):
    if size is None:
        size = get_dataset_size(config)
    molecule_list = generate_molecule_samples(model, size)
    if eval:
        folder_name = get_evaluation_set_path(config)
    else:
        folder_name = f"{config['task']}_generated_{config['graph_kernel']}"
    folder_path = os.path.join("data", folder_name)

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.join(folder_path, "raw")
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    save_molecule_list_to_dir(molecule_list, model, folder_path)


def get_evaluation_set(model, config, size):
    folder_name = get_evaluation_set_path(config)
    path = os.path.join("data", folder_name)
    if not os.path.isdir(path):
        print("Evaluation set has not been previously generated, now generating")
        create_fake_molecule_dataset(model, config, size, eval=True)
        assert os.path.isdir(path), "Created dataset but path still doesn't exist"
    new_config = config.copy()
    new_config["gen_data_path"] = path
    data = dataset.GeneratedDataset(new_config)
    return data.make_list()


@torch.no_grad()
def evaluate_molecule_samples(model, config, val=False, logger=None, step=None):
    if logger is not None:
        assert step is not None, "Not providing a step for logging"
    model.eval()
    if val:
        num_samples_to_generate = 1000
    else:
        num_samples_to_generate = config["num_samples"]
    if str(model.device) == 'cpu':
        num_samples_to_generate = 2
    if val:
        molecule_list = generate_molecule_samples(model, num_samples_to_generate)
    else:
        print("Obtaining set to evaluate")
        molecule_list = get_evaluation_set(model, config, num_samples_to_generate)
        assert len(molecule_list) == num_samples_to_generate
    print("About to compute metrics")
    sampling_metric = metrics.SamplingMolecularMetrics(model.get_dataset_info(), None).to(model.device)
    computed_metrics, smiles = sampling_metric(molecule_list, logger=logger, step=step, device=config["device"])
    if not val:
        file_path = get_evaluation_set_path(config)
        with open(os.path.join("data", file_path, "metrics.txt"), "w") as f:
            f.write(json.dumps(computed_metrics))
        with open(os.path.join("data", file_path, "smiles.txt"), "w") as f:
            for smile in smiles:
                f.write("%s\n" % smile)
    return computed_metrics


@torch.no_grad()
def evaluate_discriminator(model, config, eval_loader, eval_loader_gen):
    print("Evaluating discriminator")
    n_eval_real = 0
    n_eval_gen = 0
    eval_losses_real = []
    eval_losses_gen = []
    loss_fn = nn.BCEWithLogitsLoss()
    # split into two for loops in case of different sizes
    for batch_i, batch_real in enumerate(eval_loader):
        batch_real = batch_real.to(config["device"])
        out_real = model(batch_real)
        loss_real = loss_fn(out_real, torch.ones_like(out_real))
        eval_losses_real.append(loss_real.item()*len(out_real))
        n_eval_real += len(out_real)
    for batch_i, batch_gen in enumerate(eval_loader_gen):
        batch_gen = batch_gen.to(config["device"])
        out_gen = model(batch_gen)
        loss_gen = loss_fn(out_gen, torch.zeros_like(out_gen))
        eval_losses_gen.append(loss_gen.item()*len(out_gen))
        n_eval_gen += len(out_gen)
    #loss_avg = (np.sum(eval_losses_real) + np.sum(eval_losses_gen))/(n_eval_gen + n_eval_real)
    loss_real_avg = np.sum(eval_losses_real)/n_eval_real
    loss_gen_avg = np.sum(eval_losses_gen)/n_eval_gen
    loss_avg = (loss_real_avg + loss_gen_avg)/2
    return loss_avg, loss_real_avg, loss_gen_avg
