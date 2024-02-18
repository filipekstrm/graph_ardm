import torch

from ..transformer import general_utils

import torch_scatter as tscatt

from ..ardm import graph_kernels


class DiscriminatorAutoregressiveGraphKernel(graph_kernels.AutoregressiveGraphKernel):
    # In this kernel, we make sure to sample number of destructive steps between 0 and num_elements - 1
    def _get_num_edges_nodes_to_mask(self, node_mask):
        num_nodes = torch.sum(node_mask, dim=-1)
        num_edges = num_nodes*(num_nodes-1)/2
        num_elements = num_edges + num_nodes
        
        num_steps = graph_kernels.sample_num_steps(num_elements - 1, 0)
        
        element_mask = general_utils.batch_topk_mask(num_elements, num_elements.max(), self.device)
        logits = self.gumbel.sample(element_mask.shape).to(self.device)
        logits = element_mask*logits + (~element_mask)*(-1e9)
        logits_sorted, indices_sorted = logits.sort(descending=True, dim=-1)

        step_mask = general_utils.batch_topk_mask(num_steps, num_elements.max(), self.device)
        
        step_mask = step_mask.gather(1, indices_sorted)
        index1, index2 = torch.where(step_mask)
        num_nodes_repeat = num_nodes.repeat_interleave(num_steps)
        node_chosen_bool = (index2 < num_nodes_repeat).float()
        t_nodes = tscatt.scatter(node_chosen_bool, index1, reduce="sum", dim_size=node_mask.shape[0])
        t_edges = num_steps - t_nodes
        
        return t_edges, t_nodes


class DiscriminatorPartialAutoregressiveGraphKernel(graph_kernels.AutoregressiveGraphKernel):
    # In this kernel, we make sure to sample number of destructive steps between 0 and % of total number of elements
    def __init__(self, config, no_model=False):
        super().__init__(config, no_model)
        self.gradual_increase = config["gradual_max_masked_percentage"] != 0
        if self.gradual_increase:
            self.max_percentage = 0.0
            self.increase_every = config["increase_masked_percentage_every"]
        else:
            self.max_percentage = config["max_masked_percentage"]
        if self.max_percentage is None:
            raise ValueError("Max masked percentage has not been set (is currently None)")
        self.step = 0

    def step_update(self, step):
        if self.gradual_increase:
            if (step % self.increase_every) == 0:
                self.max_percentage = min(1.0, self.max_percentage + 0.05)
        #wandb.log({"max_masked_percentage": self.max_percentage}, step=self.step, commit=False)
        return
    
    def get_max_percentage(self):
        return self.max_percentage

    def _get_num_edges_nodes_to_mask(self, node_mask):
        num_nodes = torch.sum(node_mask, dim=-1)
        num_edges = num_nodes * (num_nodes - 1) / 2
        num_elements = num_edges + num_nodes

        num_steps = graph_kernels.sample_num_steps(((num_elements - 1)*self.max_percentage).round().long(), 0)

        element_mask = general_utils.batch_topk_mask(num_elements, num_elements.max(), self.device)
        logits = self.gumbel.sample(element_mask.shape).to(self.device)
        logits = element_mask * logits + (~element_mask) * (-1e9)
        logits_sorted, indices_sorted = logits.sort(descending=True, dim=-1)

        step_mask = general_utils.batch_topk_mask(num_steps, num_elements.max(), self.device)

        step_mask = step_mask.gather(1, indices_sorted)
        index1, index2 = torch.where(step_mask)
        num_nodes_repeat = num_nodes.repeat_interleave(num_steps)
        node_chosen_bool = (index2 < num_nodes_repeat).float()
        t_nodes = tscatt.scatter(node_chosen_bool, index1, reduce="sum", dim_size=node_mask.shape[0])
        t_edges = num_steps - t_nodes

        return t_edges, t_nodes


class DiscriminatorNodeEdgesNodeGraphKernel(graph_kernels.NodeEdgesNodeGraphKernel):
    # For this kernel, the first step is always to generate node
    # Therefore, we never mask all nodes, but keep at least one
    # We also want the case that all elements are unmasked
    # Therefore, it is always ok to mask 0 edges, even when 0 nodes have been masked
    def _get_num_nodes_to_mask(self, node_mask):
        num_nodes = torch.sum(node_mask, dim=-1)
        t_nodes = graph_kernels.sample_num_steps(num_nodes - 1, min_steps=0)  # keep at least one unmasked node
        return t_nodes, num_nodes

    def _num_edges_to_mask(self, num_nodes, t_nodes):
        num_possible_edges = num_nodes - t_nodes - 1  # -1 to remove self-loops
        num_edges_to_mask = graph_kernels.sample_num_steps(num_possible_edges, 0)  # masking 0 edges is always ok
        no_edges_mask = (num_edges_to_mask == 0).squeeze()
        return num_edges_to_mask, no_edges_mask


class DiscriminatorNodesBeforeEdgesGraphKernel(graph_kernels.NodesBeforeEdgesGraphKernel):
    # This kernel is adjusted such that the step is sampled as before, but adding 1
    def sample_t(self, node_mask):
        # t_nodes and t_edges are the number of nodes/edges to mask
        num_nodes = self._num_nodes(node_mask)
        num_edges = (num_nodes*(num_nodes - 1)/2).long()
        total_num_elements = num_nodes + num_edges
        k_prob = general_utils.batch_topk_mask(total_num_elements, total_num_elements.max(), self.device).float()
        k = k_prob.multinomial(1).squeeze() + 1  # k is number of unmasked elements, + 1 for discriminator training
        t_nodes = (num_nodes - k)*(k < num_nodes)
        t_edges = num_edges*(k < num_nodes) + (total_num_elements - k)*(k >= num_nodes)
        return t_nodes, t_edges
