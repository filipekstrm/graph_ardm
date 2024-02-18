import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformer import general_utils, transformer

from ..dataset import qm9_dataset, moses_dataset

import torch_scatter as tscatt

NODE_EDGE_OUTPUT_DIMS = {"qm9": (5, 5), "sbm": (1, 2), "planar": (1, 2), "community": (1, 2),
                         "qm9_no_h": (4, 5), "moses": (8, 5)}

dataset_infos = {
    "qm9": qm9_dataset.QM9infos,
    "qm9_no_h": qm9_dataset.QM9infos,
    "moses": moses_dataset.MOSESinfos
}


class GraphKernel(nn.Module):
    def __init__(self, config, no_model=False):
        super().__init__()
        self.device = config['device']
        if not no_model:
            self.model = transformer.GraphTransformerWrapper(config)
        self.gumbel = torch.distributions.Gumbel(0, 1)
        self.datasetinfo = dataset_infos[config["task"]](config)

    def _create_graph_list(self, data, num_nodes):
        Xclass = torch.argmax(data.X, dim=-1)
        Eclass = torch.argmax(data.E, dim=-1)

        graph_list = []
        for i, n in enumerate(num_nodes):
            node_features = Xclass[i, :n]
            edge_features = Eclass[i, :n, :n]
            graph_list.append((node_features, edge_features))
        return graph_list


class AbsorbingGraphKernel(GraphKernel):
    def __init__(self, config, no_model=False, same_order=False):
        super().__init__(config, no_model)
        self.n_distr = torch.distributions.Categorical(probs=self.datasetinfo.n_nodes)
        self.same_order = same_order

        self.node_output_dim, self.edge_output_dim = NODE_EDGE_OUTPUT_DIMS[config["task"]]

    def get_noisy_data(self, X, E, y, node_mask):
        raise NotImplementedError

    def _nodes_to_unmask_map(self, X, E):
        raise NotImplementedError

    def _unmask_nodes(self, X, E, logits_x, step_size):
        absorbed_nodes, num_absorbed_nodes = self._nodes_to_unmask_map(X, E)
        num_nodes_to_unmask = torch.minimum(num_absorbed_nodes, step_size * torch.ones_like(num_absorbed_nodes))
        if num_nodes_to_unmask.max() < 1:
            return X
        # else

        Xclass = torch.argmax(X, dim=-1)

        # first, draw which nodes to unmask
        logits_nodes = -1 * (~absorbed_nodes) * 1e9
        logits_nodes[absorbed_nodes] = self.gumbel.sample(logits_nodes[absorbed_nodes].shape).to(self.device)
        _, sorted_indices = logits_nodes.sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(num_nodes_to_unmask, logits_nodes.shape[1], self.device)
        nodes_to_unmask = topk.gather(-1, sorted_indices.argsort())
        if self.same_order:
            nodes_to_unmask = nodes_to_unmask[0].unsqueeze(0).repeat(X.shape[0], 1)
        assert torch.all(torch.logical_and(nodes_to_unmask, absorbed_nodes).sum(dim=-1) == num_nodes_to_unmask)

        # second, predict class for all nodes (even if already unmasked)
        logits_x_tilde = logits_x.clone()
        logits_x_tilde[nodes_to_unmask] = logits_x_tilde[nodes_to_unmask] + self.gumbel.sample(
            logits_x_tilde[nodes_to_unmask].shape).to(self.device)
        pred_class = torch.argmax(logits_x_tilde, dim=-1)

        # Now, replace class values with predicted classes for nodes selected to unmask
        Xclass = (~nodes_to_unmask) * Xclass + nodes_to_unmask * pred_class

        # Make onehot
        X = F.one_hot(Xclass, num_classes=self.node_output_dim + 1).float()
        return X

    def _edges_to_unmask_map(self, X, E):
        raise NotImplementedError

    def _get_masked_nodes_map(self, X):
        return torch.argmax(X, dim=-1) == self.node_output_dim

    def _get_padded_nodes_map(self, X):
        return torch.sum(X, dim=-1) == 0

    def _get_padded_edges_map(self, E):
        return torch.max(E, dim=-1)[0] < 1

    def _get_unmasked_nodes_map(self, X):
        masked_nodes_map = self._get_masked_nodes_map(X)
        padding_map = self._get_padded_nodes_map(X)
        return torch.logical_and(~masked_nodes_map, ~padding_map)

    def _get_num_masked_nodes(self, X):
        return torch.sum(self._get_masked_nodes_map(X), dim=-1)

    def _get_num_unmasked_nodes(self, X):
        return torch.sum(self._get_unmasked_nodes_map(X), dim=-1)

    def _get_masked_edges_map(self, E):
        E_class = E.argmax(dim=-1)
        return E_class == (E.shape[-1] - 1)

    def _get_num_masked_edges(self, E):
        E_class = E.argmax(dim=-1)
        E_class = E_class.triu(1)  # undirected
        return torch.sum(E_class == (E.shape[-1] - 1), dim=(-1, -2))

    def _get_unmasked_edges_map(self, E):
        masked_edges = self._get_masked_edges_map(E)
        padded_edges = self._get_padded_edges_map(E)
        return torch.logical_and(~masked_edges, ~padded_edges)

    def _unmask_edges(self, X, E, logits_e, step_size):
        """
        E: b,n,n,d
        """
        bs = E.shape[0]
        n = E.shape[1]
        absorbed_edge_class, Eclass, absorbed_edges, num_absorbed_edges = self._edges_to_unmask_map(X, E)
        # num_edges_to_unmask = torch.minimum(num_absorbed_edges, step_size*torch.ones_like(num_absorbed_edges)) # b,n
        num_edges_to_unmask = torch.minimum(num_absorbed_edges, step_size)  # b,n
        if num_edges_to_unmask.max() < 1:
            return E

        # first determine which edges to unmask
        logits_edges = -1 * (~absorbed_edges) * 1e9
        logits_edges[absorbed_edges] = self.gumbel.sample(logits_edges[absorbed_edges].shape).to(self.device)
        logits_edges = logits_edges.reshape((bs, -1))  # b, 2*n
        _, sorted_indices = logits_edges.sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(num_edges_to_unmask, logits_edges.shape[1], self.device)
        edges_to_unmask = topk.gather(-1, sorted_indices.argsort())  # b, 2*n
        edges_to_unmask = edges_to_unmask.reshape((bs, n, n))
        if self.same_order:
            edges_to_unmask = edges_to_unmask[0].unsqueeze(0).repeat(bs, 1, 1)
        assert torch.all(torch.logical_and(edges_to_unmask, absorbed_edges).sum(dim=(-1, -2)) == num_edges_to_unmask)

        # now, predicting all edges
        logits_e_tilde = logits_e.clone()
        logits_e_tilde[edges_to_unmask] = logits_e_tilde[edges_to_unmask] + self.gumbel.sample(
            logits_e_tilde[edges_to_unmask].shape).to(self.device)
        pred_class = torch.argmax(logits_e_tilde, dim=-1)  # b,n,n

        # Replace class with predicted class for edges to be unmasked
        Eclass = (~edges_to_unmask) * Eclass + edges_to_unmask * pred_class
        Eclass = Eclass.triu(1)
        Eclass = Eclass + Eclass.transpose(1, 2)

        # Make onehot
        Eonehot = F.one_hot(Eclass, num_classes=absorbed_edge_class)
        E = general_utils.encode_no_edge(Eonehot.float())
        return E

    def _get_num_masked_nodes(self, X):
        return torch.sum(torch.argmax(X, dim=-1) == self.node_output_dim, dim=-1)

    def _get_num_masked_edges(self, E):
        E_class = E.argmax(dim=-1)
        E_class = E_class.triu(1)  # undirected

        return torch.sum(E_class == (E.shape[-1] - 1), dim=(-1, -2))

    def _get_num_masked_elements(self, X, E):
        return self._get_num_masked_nodes(X) + self._get_num_masked_edges(E)

    def _get_num_edges_and_nodes_to_unmask(self, X, E, y, node_mask, step_size=1):
        raise NotImplementedError

    def _num_nodes(self, node_mask):
        return torch.sum(node_mask, dim=-1)

    def _gumbel_mask_edges(self, e_class, t_edges, potential_edges_to_mask, num_classes):
        bs = e_class.shape[0]
        max_num_nodes = e_class.shape[1]
        # Gumbel top-k trick to sample edges
        logits = self.gumbel.sample(e_class.shape).to(self.device)
        logits = potential_edges_to_mask * logits + (~potential_edges_to_mask) * (-1e9)

        logits_sorted, indices_sorted = logits.reshape((bs, -1)).sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(t_edges, logits_sorted.shape[1], self.device)

        # Sort e_class according to logits and replace topk with absorbed class
        e_class_sorted = e_class.reshape((bs, -1)).gather(1, indices_sorted)
        e_class_sorted[topk] = num_classes - 1

        # Revert sorting, but now with the absorbed edges
        e_class = e_class_sorted.gather(-1, indices_sorted.argsort(-1))
        e_class = e_class.reshape((bs, max_num_nodes, max_num_nodes))
        # Undirected graph
        e_class = e_class.triu(1)
        e_class = e_class + e_class.transpose(1, 2)
        return e_class
    
    def forward_step(self, X, E, y, node_mask, step_size=1):
        num_edges_to_unmask, num_nodes_to_unmask = self._get_num_edges_and_nodes_to_unmask(X, E, y, node_mask,
                                                                                           step_size)

        if self.same_order:
            num_edges_to_unmask = num_edges_to_unmask[0].repeat(X.shape[0])
            num_nodes_to_unmask = num_nodes_to_unmask[0].repeat(X.shape[0])

        logits = self.model(X, E, y, node_mask)

        logits_e = logits.E
        Enew = self._unmask_edges(X, E, logits_e, num_edges_to_unmask)
        num_masked_edges_diff = self._get_num_masked_edges(E) - self._get_num_masked_edges(Enew)
        assert torch.all(
            num_masked_edges_diff == num_edges_to_unmask), f"\n{num_edges_to_unmask} \n{num_masked_edges_diff}"

        logits_x = logits.X
        Xnew = self._unmask_nodes(X, E, logits_x, num_nodes_to_unmask)
        num_masked_nodes_diff = self._get_num_masked_nodes(X) - self._get_num_masked_nodes(Xnew)
        assert torch.all(
            num_masked_nodes_diff == num_nodes_to_unmask), f"\n{num_nodes_to_unmask} \n{num_masked_nodes_diff}"
        new_data = general_utils.PlaceHolder(X=Xnew, E=Enew, y=y).mask(node_mask)
        return new_data

    def sample_p0_given_n(self, batch_size, n):
        num_nodes = n * torch.ones(batch_size)
        X = torch.zeros((batch_size, n, self.node_output_dim + 1), device=self.device)
        X[..., -1] = 1
        E = torch.zeros((batch_size, n, n, self.edge_output_dim + 1), device=self.device)
        E[..., -1] = 1
        E = general_utils.encode_no_edge(E)
        y = torch.zeros((batch_size, 1), device=self.device)
        node_mask = torch.ones((batch_size, n), device=self.device).bool()
        new_data = general_utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
        return num_nodes.long(), new_data, node_mask

    def generate_from_prior(self, batch_size):
        n_vec = self.n_distr.sample((batch_size,))
        n_values, num_samples_vec = torch.unique(n_vec, return_counts=True)
        new_data_list = []
        num_nodes_list = []
        node_mask_list = []
        for n, num_samples in zip(n_values, num_samples_vec):
            num_nodes, new_data, node_mask = self.sample_p0_given_n(num_samples, n)
            new_data_list.append(new_data)
            num_nodes_list.append(num_nodes)
            node_mask_list.append(node_mask)

        return num_nodes_list, new_data_list, node_mask_list

    def sample(self, new_data, node_mask):
        while torch.any(self._get_num_masked_nodes(new_data.X) != 0) or torch.any(self._get_num_masked_edges(new_data.E) != 0):
            new_data = self.forward_step(new_data.X, new_data.E, new_data.y, node_mask, 1)

        new_data.X = new_data.X[..., :self.node_output_dim]
        assert torch.all(torch.logical_xor(torch.any(new_data.X == 1, dim=-1), ~node_mask))
        new_data.E = new_data.E[..., :self.edge_output_dim]
        assert torch.all(torch.sum(new_data.E==1, dim=(-1, -2, -3)) == torch.sum(node_mask, dim=-1)**2 - torch.sum(node_mask, dim=-1))
        return new_data

    def generate(self, num_samples_to_generate):
        print(f'Generating in total {num_samples_to_generate} molecules')
        graphs = []
        st = time.time()
        num_nodes_list, new_data_list, node_mask_list = self.generate_from_prior(num_samples_to_generate)
        for j, (new_data, node_mask, num_nodes) in enumerate(zip(new_data_list, node_mask_list, num_nodes_list)):
            assert int(num_nodes.max()) == int(num_nodes.min()), num_nodes
            print(f"Generating {node_mask.shape[0]} molecules with {num_nodes[0]} nodes")
            new_data = self.sample(new_data, node_mask)
            graphs.extend(self._create_graph_list(new_data, num_nodes))
        print(f"Done with generation, took {time.time() - st} s")
        return graphs


def sample_num_steps(max_steps, min_steps=1):
    return ((max_steps - min_steps + 1) * torch.rand(len(max_steps), device=max_steps.device) - 0.5).round().long() + min_steps


class AutoregressiveGraphKernel(AbsorbingGraphKernel):
    def __init__(self, config, no_model=False):
        super().__init__(config, no_model)

    def _get_num_edges_nodes_to_mask(self, node_mask):
        num_nodes = torch.sum(node_mask, dim=-1)
        num_edges = num_nodes*(num_nodes-1)/2
        num_elements = num_edges + num_nodes
        
        num_steps = sample_num_steps(num_elements)
        
        element_mask = general_utils.batch_topk_mask(num_elements, num_elements.max(), self.device)
        logits = self.gumbel.sample(element_mask.shape).to(self.device)
        logits = element_mask*logits + (~element_mask)*(-1e9)
        logits_sorted, indices_sorted = logits.sort(descending=True, dim=-1)

        step_mask = general_utils.batch_topk_mask(num_steps, num_elements.max(), self.device)
        
        step_mask = step_mask.gather(1, indices_sorted)
        index1, index2 = torch.where(step_mask)
        num_nodes_repeat = num_nodes.repeat_interleave(num_steps)
        node_chosen_bool = (index2 < num_nodes_repeat).float()
        t_nodes = tscatt.scatter(node_chosen_bool, index1, reduce="sum").long()
        t_edges = num_steps - t_nodes
        
        return t_edges, t_nodes
        
    def get_noisy_data(self, X, E, y, node_mask):
        t_edges, t_nodes = self._get_num_edges_nodes_to_mask(node_mask)
        bs = X.shape[0]
        max_num_nodes = node_mask.shape[1]
        num_classes = E.shape[-1] + 1  # include the masked class

        # masking edges
        e_class = E.argmax(dim=-1)  # Some ambiguity as a 0 can mean no edge or padding or self-loop
        actual_edges = E.sum(dim=-1) == 1  # Get map of actual edges (i.e., self-loops and padding is set to False)
        actual_edges = actual_edges.triu(1)  # undirected graph so only working on one half right now

        # Gumbel top-k trick to sample edges
        logits = self.gumbel.sample(e_class.shape).to(self.device)
        logits = actual_edges * logits + (~actual_edges) * (-1e9)

        logits_sorted, indices_sorted = logits.reshape((bs, -1)).sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(t_edges, logits_sorted.shape[1], self.device)

        # Sort e_class according to logits and replace topk with absorbed class
        e_class_sorted = e_class.reshape((bs, -1)).gather(1, indices_sorted)
        e_class_sorted[topk] = num_classes - 1

        # Revert sorting, but now with the absorbed edges
        e_class = e_class_sorted.gather(-1, indices_sorted.argsort(-1))
        e_class = e_class.reshape((bs, max_num_nodes, max_num_nodes))
        # Undirected graph
        e_class = e_class.triu(1)
        e_class = e_class + e_class.transpose(1, 2)

        absorbed_edges = e_class == (num_classes - 1)
        absorbed_edges = absorbed_edges.reshape((bs, -1))

        E = general_utils.encode_no_edge(F.one_hot(e_class, num_classes=num_classes).float())

        # mask nodes
        X = X[..., :self.node_output_dim]
        X = torch.cat([X, torch.zeros((bs, max_num_nodes, 1), device=self.device)], dim=-1)
        logits = self.gumbel.sample(X.shape[0:2]).to(self.device)
        logits = node_mask * logits + (~node_mask) * (-1e9)
        logits_sorted, indices_sorted = logits.sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(t_nodes, logits_sorted.shape[1], self.device)

        index1 = torch.arange(bs, device=self.device).long().repeat_interleave(t_nodes.long())
        index2 = indices_sorted[topk].flatten()
        X[index1, index2] = torch.tensor([0] * self.node_output_dim + [1], device=self.device).float()

        absorbed_nodes = torch.zeros(X.shape[0:2], device=self.device)
        absorbed_nodes[index1, index2] = 1
        absorbed_nodes = absorbed_nodes.bool()

        # mask y
        y = torch.zeros((X.shape[0], 1), device=self.device)
        noisy_data = general_utils.PlaceHolder(X, E, y).mask(node_mask)
        return noisy_data, {'abs_edges': absorbed_edges, 'abs_nodes': absorbed_nodes}

    def _edges_to_unmask_map(self, X, E):
        bs = E.shape[0]
        absorbed_edge_class = E.shape[-1]
        Eclass = E.argmax(-1)  # b,n,n
        absorbed_edges = Eclass == absorbed_edge_class - 1  # b,n,n
        absorbed_edges = absorbed_edges.triu(1)  # To ensure undirected
        num_absorbed_edges = absorbed_edges.reshape((bs, -1)).sum(dim=-1)  # b

        return absorbed_edge_class, Eclass, absorbed_edges, num_absorbed_edges

    def _nodes_to_unmask_map(self, X, E):
        absorbed_nodes = torch.argmax(X, dim=-1) == self.node_output_dim
        num_absorbed_nodes = absorbed_nodes.sum(dim=-1)
        return absorbed_nodes, num_absorbed_nodes

    def _get_num_edges_and_nodes_to_unmask(self, X, E, y, node_mask, step_size=1):
        num_nodes = torch.sum(node_mask, dim=-1)
        num_edges = num_nodes * (num_nodes - 1) / 2
        num_masked_nodes = self._get_num_masked_nodes(X)
        num_masked_edges = self._get_num_masked_edges(E)
        bern_prob_node = num_masked_nodes / (num_masked_nodes + num_masked_edges + 1e-15)
        # TODO: this is only correct for step_size = 1, which is what I currently use. But needs fix
        binom = torch.distributions.Binomial(step_size, probs=bern_prob_node)
        num_nodes_to_unmask = binom.sample()
        num_edges_to_unmask = step_size - num_nodes_to_unmask
        num_edges_to_unmask[num_masked_edges == 0] = 0

        return num_edges_to_unmask, num_nodes_to_unmask


class NodeEdgesNodeGraphKernel(AbsorbingGraphKernel):
    def __init__(self, config, no_model=False):
        super().__init__(config, no_model)

    def _get_num_nodes_to_mask(self, node_mask):
        num_nodes = torch.sum(node_mask, dim=-1)
        t_nodes = sample_num_steps(num_nodes, min_steps=0)
        return t_nodes, num_nodes

    def _mask_nodes(self, X, node_mask):
        bs = X.shape[0]
        max_num_nodes = node_mask.shape[1]

        # mask nodes
        t_nodes, num_nodes = self._get_num_nodes_to_mask(node_mask)
        X = X[..., :self.node_output_dim]
        X = torch.cat([X, torch.zeros((bs, max_num_nodes, 1), device=self.device)], dim=-1)
        logits = self.gumbel.sample(X.shape[0:2]).to(self.device)
        logits = node_mask * logits + (~node_mask) * (-1e9)
        logits_sorted, indices_sorted = logits.sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(t_nodes, logits_sorted.shape[1], self.device)

        index1 = torch.arange(bs, device=self.device).long().repeat_interleave(t_nodes)
        index2 = indices_sorted[topk].flatten()
        X[index1, index2] = torch.tensor([0] * self.node_output_dim + [1], device=self.device).float()

        absorbed_nodes = torch.zeros(X.shape[0:2], device=self.device)
        absorbed_nodes[index1, index2] = 1
        absorbed_nodes = absorbed_nodes.bool()
        return X, t_nodes, absorbed_nodes, index1, index2

    def _num_edges_to_mask(self, num_nodes, t_nodes):
        num_possible_edges = num_nodes - t_nodes - 1  # -1 to remove self-loops
        min_edges = (t_nodes == 0).long()
        num_edges_to_mask = sample_num_steps(num_possible_edges, min_edges)
        no_edges_mask = (num_edges_to_mask == 0).squeeze()
        return num_edges_to_mask, no_edges_mask

    def _latest_sampled_node(self, absorbed_nodes, node_mask, no_edges_map, bs):
        prob = torch.logical_and(~absorbed_nodes, node_mask).float()
        prob[no_edges_map] = 1  # again to avoid errors
        index1 = torch.arange(bs, device=self.device)
        index2 = prob.multinomial(1).flatten()
        return index1[~no_edges_map], index2[~no_edges_map]

    def _update_potential_edges_to_mask(self, current_map, index1, index2):
        temp_map = torch.zeros_like(current_map)
        temp_map[index1, index2, :] = True
        temp_map[index1, :, index2] = True
        return torch.logical_and(current_map, temp_map)

    def _mask_edges(self, E, t_nodes, absorbed_nodes, node_mask, index1, index2):
        # setup
        bs = E.shape[0]
        num_nodes = self._num_nodes(node_mask)
        num_classes = E.shape[-1] + 1  # include the masked class

        # Get map of actual edges (i.e., self-loops and padding is set to False)
        actual_edges = E.sum(dim=-1) == 1
        e_class = E.argmax(dim=-1)  # Some ambiguity as a 0 can mean no edge or padding or self-loop

        # Mask all edges to/from masked nodes
        e_class[index1, index2, :] = num_classes - 1
        e_class[index1, :, index2] = num_classes - 1

        # Now get an initial map of which edges that could potentially be masked
        potential_edges_to_mask = torch.logical_and(actual_edges, e_class != num_classes - 1)
        # make undirected
        potential_edges_to_mask = potential_edges_to_mask.triu(1)
        e_class = e_class.triu(1)

        # get number of edges to mask, and a map over nodes where 0 edges will be masked
        t_edges, no_edges_map = self._num_edges_to_mask(num_nodes, t_nodes)

        # get nodes b, i which are the indices of the latest sampled nodes
        index1e, index2e = self._latest_sampled_node(absorbed_nodes, node_mask, no_edges_map, bs)

        # in potential_edge_to_mask, only allow edges to/from latest_sampled_node
        potential_edges_to_mask = self._update_potential_edges_to_mask(potential_edges_to_mask, index1e, index2e)

        e_class = self._gumbel_mask_edges(e_class, t_edges, potential_edges_to_mask, num_classes)

        # Set absorbed_edges
        absorbed_edges = torch.logical_and(e_class == (num_classes - 1), potential_edges_to_mask)

        absorbed_edges = absorbed_edges.reshape((bs, -1))

        E = general_utils.encode_no_edge(F.one_hot(e_class, num_classes=num_classes).float())
        assert torch.allclose(t_edges, absorbed_edges.sum(-1)), "t_edges and absorbed edges not the same"

        return E, absorbed_edges, no_edges_map

    def get_noisy_data(self, X, E, y, node_mask):
        X, t_nodes, absorbed_nodes, index1, index2 = self._mask_nodes(X, node_mask)
        E, absorbed_edges, no_edges_mask = self._mask_edges(E, t_nodes, absorbed_nodes, node_mask, index1, index2)

        # Set absorbed_nodes correct
        absorbed_nodes[~no_edges_mask] = False

        # mask y
        y = torch.zeros((X.shape[0], 1), device=self.device)
        noisy_data = general_utils.PlaceHolder(X, E, y).mask(node_mask)
        return noisy_data, {'abs_edges': absorbed_edges, 'abs_nodes': absorbed_nodes}

    def _nodes_to_unmask_map(self, X, E):
        unmasked_nodes_map = self._get_unmasked_nodes_map(X)
        num_unmasked_nodes = self._get_num_unmasked_nodes(X)
        num_unmasked_nodes_repeat = num_unmasked_nodes.repeat_interleave(num_unmasked_nodes)
        unmasked_edges_map = self._get_unmasked_edges_map(E)
        num_unmasked_edges_per_node = torch.sum(unmasked_edges_map, dim=-1)
        batch = torch.arange(X.shape[0], device=self.device)
        batch = batch.repeat_interleave(num_unmasked_nodes)
        node_next = tscatt.scatter(
            (num_unmasked_edges_per_node[unmasked_nodes_map] == (num_unmasked_nodes_repeat - 1)).float(), batch,
            dim_size=X.shape[0], reduce='mean')

        node_next[num_unmasked_nodes == 0] = 1.0  # fix corner case when there are no unmasked nodes
        node_next = node_next == 1
        node_next = node_next.unsqueeze(1)
        nodes_to_unmask = torch.logical_and(node_next, self._get_masked_nodes_map(X))
        num_nodes_to_unmask = nodes_to_unmask.sum(dim=-1)
        return nodes_to_unmask, num_nodes_to_unmask

    def _edges_to_unmask_map(self, X, E):
        absorbed_edge_class = E.shape[-1]

        bs = X.shape[0]
        Eclass = E.argmax(-1)
        unmasked_nodes_map = self._get_unmasked_nodes_map(X)
        num_unmasked_nodes = self._get_num_unmasked_nodes(X)
        batch = torch.arange(X.shape[0], device=self.device)
        batch = batch.repeat_interleave(num_unmasked_nodes)

        unmasked_edges_map = self._get_unmasked_edges_map(E)
        num_unmasked_edges_per_node = torch.sum(unmasked_edges_map, dim=-1)

        min_num_unmasked_edges = tscatt.scatter(num_unmasked_edges_per_node[unmasked_nodes_map], batch,
                                                dim_size=X.shape[0], reduce="min")
        nodes_w_edges_to_unmask_map = min_num_unmasked_edges.unsqueeze(1) == num_unmasked_edges_per_node
        nodes_w_edges_to_unmask_map = torch.logical_and(nodes_w_edges_to_unmask_map, unmasked_nodes_map)
        index1, index2 = torch.where(nodes_w_edges_to_unmask_map)
        edges_to_unmask_map = torch.zeros_like(unmasked_edges_map).bool()
        edges_to_unmask_map[index1, index2, :] = True
        edges_to_unmask_map[index1, :, index2] = True
        edges_to_unmask_map = torch.logical_and(edges_to_unmask_map, unmasked_nodes_map.unsqueeze(-1))
        edges_to_unmask_map = torch.logical_and(edges_to_unmask_map, unmasked_nodes_map.unsqueeze(-2))
        edges_to_unmask_map = torch.logical_and(edges_to_unmask_map, ~unmasked_edges_map)
        edges_to_unmask_map = edges_to_unmask_map.triu(1)
        num_edges_to_unmask = edges_to_unmask_map.reshape((bs, -1)).sum(dim=-1)  # b
        return absorbed_edge_class, Eclass, edges_to_unmask_map, num_edges_to_unmask

    def _get_num_edges_and_nodes_to_unmask(self, X, E, y, node_mask, step_size=1):
        node_to_unmask_map, _ = self._nodes_to_unmask_map(X, E)
        num_nodes_to_unmask = torch.any(node_to_unmask_map, dim=-1).long()  # we alway sample 1 at a time

        num_masked_edges = self._get_num_masked_edges(E)
        num_edges_to_unmask = 1 - num_nodes_to_unmask  # if we dont sample a node, we sample an edge
        num_edges_to_unmask[num_masked_edges == 0] = 0  # unless there are no edges left

        return num_edges_to_unmask, num_nodes_to_unmask


class NodesBeforeEdgesGraphKernel(AbsorbingGraphKernel):
    def __init__(self, config, no_model=False):
        super().__init__(config, no_model)

    def _mask_nodes(self, X, node_mask, t_nodes):
        bs = X.shape[0]
        max_num_nodes = X.shape[1]
        X = X[..., :self.node_output_dim]
        X = torch.cat([X, torch.zeros((bs, max_num_nodes, 1), device=self.device)], dim=-1)
        logits = self.gumbel.sample(X.shape[0:2]).to(self.device)
        logits = node_mask * logits + (~node_mask) * (-1e9)
        logits_sorted, indices_sorted = logits.sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(t_nodes, logits_sorted.shape[1], self.device)

        index1 = torch.arange(bs, device=self.device).long().repeat_interleave(t_nodes)
        index2 = indices_sorted[topk].flatten()
        X[index1, index2] = torch.tensor([0] * self.node_output_dim + [1], device=self.device).float()

        absorbed_nodes = torch.zeros(X.shape[0:2], device=self.device)
        absorbed_nodes[index1, index2] = 1
        absorbed_nodes = absorbed_nodes.bool()
        return X, absorbed_nodes

    def sample_t(self, node_mask):
        # t_nodes and t_edges are the number of nodes/edges to mask
        num_nodes = self._num_nodes(node_mask)
        num_edges = (num_nodes*(num_nodes - 1)/2).long()
        total_num_elements = num_nodes + num_edges
        k_prob = general_utils.batch_topk_mask(total_num_elements, total_num_elements.max(), self.device).float()
        k = k_prob.multinomial(1).squeeze()  # k is number of unmasked elements
        t_nodes = (num_nodes - k)*(k<num_nodes)
        t_edges = (num_edges)*(k<num_nodes) + (total_num_elements - k)*(k>=num_nodes)
        return t_nodes, t_edges

    def _mask_edges(self, E, t_edges):
        bs = E.shape[0]
        num_classes = E.shape[-1] + 1
        max_num_nodes = E.shape[1]
        e_class = E.argmax(dim=-1)  # Some ambiguity as a 0 can mean no edge or padding or self-loop
        actual_edges = E.sum(dim=-1) == 1  # Get map of actual edges (i.e., self-loops and padding is set to False)
        actual_edges = torch.logical_and(actual_edges, t_edges.reshape(-1, 1, 1) > 0)
        actual_edges = actual_edges.triu(1)  # undirected graph so only working on one half right now

        # Gumbel top-k trick to sample edges
        logits = self.gumbel.sample(e_class.shape).to(self.device)
        logits = actual_edges * logits + (~actual_edges) * (-1e9)

        logits_sorted, indices_sorted = logits.reshape((bs, -1)).sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(t_edges, logits_sorted.shape[1], self.device)

        # Sort e_class according to logits and replace topk with absorbed class
        e_class_sorted = e_class.reshape((bs, -1)).gather(1, indices_sorted)
        e_class_sorted[topk] = num_classes - 1

        # Revert sorting, but now with the absorbed edges
        e_class = e_class_sorted.gather(-1, indices_sorted.argsort(-1))
        e_class = e_class.reshape((bs, max_num_nodes, max_num_nodes))
        # Undirected graph
        e_class = e_class.triu(1)
        e_class = e_class + e_class.transpose(1, 2)

        absorbed_edges = e_class == (num_classes - 1)
        absorbed_edges = absorbed_edges.reshape((bs, -1))

        E = general_utils.encode_no_edge(F.one_hot(e_class, num_classes=num_classes).float())

        return E, absorbed_edges

    def get_noisy_data(self, X, E, y, node_mask):
        t_nodes, t_edges = self.sample_t(node_mask)

        # mask edges
        X, absorbed_nodes = self._mask_nodes(X, node_mask, t_nodes)

        # masking edges
        E, absorbed_edges = self._mask_edges(E, t_edges)

        # mask y
        y = torch.zeros((X.shape[0], 1), device=self.device)
        noisy_data = general_utils.PlaceHolder(X, E, y).mask(node_mask)
        return noisy_data, {'abs_edges': absorbed_edges, 'abs_nodes': absorbed_nodes}

    def _nodes_to_unmask_map(self, X, E):
        masked_nodes = self._get_masked_nodes_map(X)
        num_masked_nodes = self._get_num_masked_nodes(X)
        return masked_nodes, num_masked_nodes

    def _edges_to_unmask_map(self, X, E):
        absorbed_edge_class = E.shape[-1]
        bs = X.shape[0]
        Eclass = E.argmax(-1)

        masked_edges = self._get_masked_edges_map(E)
        edges_next = self._get_num_masked_nodes(X) == 0
        edges_to_unmask_map = torch.logical_and(masked_edges, edges_next.reshape(-1, 1, 1))
        edges_to_unmask_map = edges_to_unmask_map.triu(1)
        num_edges_to_unmask = edges_to_unmask_map.reshape((bs, -1)).sum(dim=-1)  # b

        return absorbed_edge_class, Eclass, edges_to_unmask_map, num_edges_to_unmask

    def _get_num_edges_and_nodes_to_unmask(self, X, E, y, node_mask, step_size=1):
        node_to_unmask_map, _ = self._nodes_to_unmask_map(X, E)
        num_nodes_to_unmask = torch.any(node_to_unmask_map, dim=-1).long()  # we alway sample 1 at a time

        num_masked_edges = self._get_num_masked_edges(E)
        num_edges_to_unmask = 1 - num_nodes_to_unmask  # if we dont sample a node, we sample an edge
        num_edges_to_unmask[num_masked_edges == 0] = 0  # unless there are no edges left

        return num_edges_to_unmask, num_nodes_to_unmask
