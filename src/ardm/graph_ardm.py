import torch
from . import graph_kernels
import torch.nn as nn

from ..transformer import general_utils

from .. import loss_functions

graph_kernels_dict = {
    'autoreg': graph_kernels.AutoregressiveGraphKernel,
    'nen': graph_kernels.NodeEdgesNodeGraphKernel,
    'ne': graph_kernels.NodesBeforeEdgesGraphKernel
}


class GraphARDM(nn.Module):
    def __init__(self, config):
        super(GraphARDM, self).__init__()
        self.device = config['device']
        self.kernel = graph_kernels_dict[config['graph_kernel']](config)
        self.edge_lambda = config['edge_lambda']
        self.node_lambda = config['node_lambda']
        self.dataset_info = graph_kernels.dataset_infos[config["task"]](config)

    @property
    def node_output_dim(self):
        return self.kernel.node_output_dim
    
    @property
    def edge_output_dim(self):
        return self.kernel.edge_output_dim
    
    def get_dataset_info(self):
        return self.dataset_info
    
    def forward(self, data):
        dense_data, node_mask = general_utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)

        noisy_data, absorbed_elements = self.kernel.get_noisy_data(dense_data.X, dense_data.E, data.y, node_mask)
        if noisy_data.X.shape[-1] != self.kernel.node_output_dim + 1:
            raise ValueError("X contains more than it should")
        if noisy_data.E.shape[-1] != self.kernel.edge_output_dim + 1:
            raise ValueError("E contains more than it should")
        out = self.kernel.model.forward(noisy_data.X, noisy_data.E, noisy_data.y, node_mask)

        # X might contain some extra unwanted features
        ground_truth_nodes = torch.argmax(dense_data.X[..., :self.kernel.node_output_dim], dim=-1)
        ground_truth_edges = torch.argmax(dense_data.E, dim=-1).flatten(1, 2)

        # TODO: should there be a triu() at the absorbed_elements mask (or any equivalent solution, e.g., division by 2)
        node_loss, edge_loss = loss_functions.graph_uniform_cross_entropy_loss({"mask": absorbed_elements["abs_nodes"],
                                                                                "logits": out.X,
                                                                                "target": ground_truth_nodes,
                                                                                "lambda": self.node_lambda,
                                                                                "D": torch.sum(node_mask, dim=-1)},
                                                                               {"mask": absorbed_elements["abs_edges"],
                                                                                "logits": out.E.flatten(1, 2),
                                                                                "target": ground_truth_edges,
                                                                                "lambda": self.edge_lambda}
                                                                               )

        logging_dict = {'train/node_loss': node_loss.item(), 'train/edge_loss': edge_loss.item()}
        return {'loss': node_loss + edge_loss, 'logging_dict': logging_dict}

    def generate(self, batch_size):
        return self.kernel.generate(batch_size)

        




