import time
import os
import csv
import torch
import torch.nn as nn
from ..transformer import transformer

from ..pf import pf_utils

import torch.nn.functional as F

from ..transformer import general_utils

from . import discriminator_graph_kernels


discriminator_graph_kernels_dict = {
    'autoreg': discriminator_graph_kernels.DiscriminatorAutoregressiveGraphKernel,
    'nen': discriminator_graph_kernels.DiscriminatorNodeEdgesNodeGraphKernel,
    'ne': discriminator_graph_kernels.DiscriminatorNodesBeforeEdgesGraphKernel,
    'partialautoreg': discriminator_graph_kernels.DiscriminatorPartialAutoregressiveGraphKernel
}

discriminator_models = {
    "transformer": transformer.TransformerDiscriminator
}
        

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.device = config["device"]
        self.network = discriminator_models[config["discriminator"]](config)
        self.kernel = discriminator_graph_kernels_dict[config["graph_kernel"]](config, no_model=True)

    def predict(self, X, E, y, node_mask):
        return self.network(X, E, y, node_mask)

    def get_noisy_data(self, X, E, y, node_mask):
        return self.kernel.get_noisy_data(X, E, y, node_mask)[0]

    def _add_dimension(self, X, E, y, node_mask):
        X = X[..., :self.kernel.node_output_dim]
        X = torch.cat([X, torch.zeros((X.shape[0], X.shape[1], 1), device=self.device)], dim=-1)
        E = torch.cat([E, torch.zeros((E.shape[0], E.shape[1], E.shape[2], 1), device=self.device)], dim=-1)
        y = torch.zeros((X.shape[0], 1), device=self.device)
        return general_utils.PlaceHolder(X, E, y).mask(node_mask)
    
    def forward(self, data):
        dense_data, node_mask = general_utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)

        noisy_data = self.get_noisy_data(dense_data.X, dense_data.E, data.y, node_mask)
        if noisy_data.X.shape[-1] != self.kernel.node_output_dim + 1:
            raise ValueError("X contains more than it should")
        if noisy_data.E.shape[-1] != self.kernel.edge_output_dim + 1:
            raise ValueError("E contains more than it should")
        return self.network(noisy_data.X, noisy_data.E, noisy_data.y, node_mask)


class DiscriminatorGuidance(nn.Module):
    def __init__(self, model, discriminator, config):
        super(DiscriminatorGuidance, self).__init__()
        self.model = model
        self.mode = self.get_mode()
        self.discriminator = discriminator
        self.device = config["device"]
        self.masked_percentage_lim = config["max_masked_percentage"]
        self.tempering = config["discriminator_tempering"] != 0
        self.temp_constant = config["discriminator_temp_constant"]

    @property
    def n_distr(self):
        return self.model.kernel.n_distr

    def sample_p0_given_n(self, batch_size, n):
        return self.model.kernel.sample_p0_given_n(batch_size, n)

    def _get_num_masked_nodes(self, X):
        return self.model.kernel._get_num_masked_nodes(X)

    def _get_num_masked_edges(self, E):
        return self.model.kernel._get_num_masked_edges(E)

    def _get_num_edges_and_nodes_to_unmask(self, X, E, y, node_mask, step_size):
        return self.model.kernel._get_num_edges_and_nodes_to_unmask(X, E, y, node_mask, step_size)

    def _nodes_to_unmask_map(self, X, E):
        return self.model.kernel._nodes_to_unmask_map(X, E)

    def _edges_to_unmask_map(self, X, E):
        return self.model.kernel._edges_to_unmask_map(X, E)

    def generate_from_prior(self, batch_size):
        return self.model.kernel.generate_from_prior(batch_size)

    def _create_graph_list(self, data, num_nodes):
        return self.model.kernel._create_graph_list(data, num_nodes)

    def _num_nodes(self, node_mask):
        return self.model.kernel._num_nodes(node_mask)

    @property
    def node_output_dim(self):
        return self.model.kernel.node_output_dim

    @property
    def edge_output_dim(self):
        return self.model.kernel.edge_output_dim

    @property
    def gumbel(self):
        return self.model.kernel.gumbel

    def forward(self, X, E, y, node_mask):
        return self.discriminator.predict(X, E, y, node_mask)
        
    def get_dataset_info(self):
        return self.model.get_dataset_info()

    def _generate(self, new_data, node_mask):
        raise NotImplementedError

    def get_mode(self):
        raise NotImplementedError

    def get_temp_exponent(self, X, E, node_mask):
        if self.tempering:
            num_nodes = torch.sum(node_mask, dim=-1)
            num_elements = num_nodes + num_nodes*(num_nodes - 1)/2
            num_masked_elements = self._get_num_masked_elements(X, E)
            temp_exponent = 1.0 - num_masked_elements/num_elements
        else:
            temp_exponent = 1.0
        return self.temp_constant*temp_exponent

    def generate(self, num_samples_to_generate):
        print(f'Generating in total {num_samples_to_generate} molecules, using {self.get_mode()}')
        graphs = []
        st = time.time()
        num_nodes_list, new_data_list, node_mask_list = self.generate_from_prior(num_samples_to_generate)
        for j, (new_data, node_mask, num_nodes) in enumerate(zip(new_data_list, node_mask_list, num_nodes_list)):
            assert int(num_nodes.max()) == int(num_nodes.min()), num_nodes
            print(f"Generating {node_mask.shape[0]} molecules with {num_nodes[0]} nodes")
            new_data, _ = self._generate(new_data, node_mask)
            graphs.extend(self._create_graph_list(new_data, num_nodes))
        print(f"Done with generation, took {time.time() - st} s")
        return graphs


class ARDG(DiscriminatorGuidance):
    def get_mode(self):
        return "ardg"

    def _unmask_edges(self, X, E, y, node_mask, logits_e, step_size):
        bs = E.shape[0]
        n = E.shape[1]
        absorbed_edge_class, Eclass, absorbed_edges, num_absorbed_edges = self._edges_to_unmask_map(X, E)
        # num_edges_to_unmask = torch.minimum(num_absorbed_edges, step_size*torch.ones_like(num_absorbed_edges)) # b,n
        num_edges_to_unmask = torch.minimum(num_absorbed_edges, step_size)  # b,n
        if num_edges_to_unmask.max() < 1:
            return E
        # Determine which edges to unmask
        logits_edges = self.gumbel.sample(E.shape[0:3]).to(self.device) - (~absorbed_edges) * 1e9  # b,n,n
        logits_edges = logits_edges.reshape((bs, -1))  # b, 2*n
        _, sorted_indices = logits_edges.sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(num_edges_to_unmask, logits_edges.shape[1], self.device)
        edges_to_unmask = topk.gather(-1, sorted_indices.argsort())  # b, 2*n
        edges_to_unmask = edges_to_unmask.reshape((bs, n, n))
        assert torch.all(
            torch.logical_and(edges_to_unmask, absorbed_edges).sum(dim=(-1, -2)) == num_edges_to_unmask)

        # Replace class with predicted class for edges to be unmasked
        w_t = torch.ones_like(E)[..., :self.edge_output_dim]
        index1, index2, index3 = torch.where(edges_to_unmask)
        new_class = torch.zeros_like(Eclass)
        for i in range(self.edge_output_dim):
            Eclass = (~edges_to_unmask) * Eclass + edges_to_unmask * new_class
            Eclass = Eclass.triu(1)
            Eclass = Eclass + Eclass.transpose(1, 2)
            Eonehot = F.one_hot(Eclass, num_classes=absorbed_edge_class)
            E_new = general_utils.encode_no_edge(Eonehot.float())
            d_logit = self.forward(X, E_new, y, node_mask).reshape(-1)
            d_logit = (d_logit*(self.get_temp_exponent(X, E, node_mask)))[index1]
            w_t[index1, index2, index3, i] = torch.exp(d_logit)
            new_class = new_class + 1
        probs_e = logits_e.softmax(-1)
        guided_probs = (probs_e * w_t).reshape((-1, self.edge_output_dim))

        pred_class = guided_probs.multinomial(1).reshape((-1, n, n))

        # Replace class with predicted class for edges to be unmasked
        Eclass = (~edges_to_unmask) * Eclass + edges_to_unmask * pred_class
        Eclass = Eclass.triu(1)
        Eclass = Eclass + Eclass.transpose(1, 2)

        # Make onehot
        Eonehot = F.one_hot(Eclass, num_classes=absorbed_edge_class)
        E = general_utils.encode_no_edge(Eonehot.float())
        return E

    def _unmask_nodes(self, X, E, y, node_mask, logits_x, step_size):
        absorbed_nodes, num_absorbed_nodes = self._nodes_to_unmask_map(X, E)
        num_nodes_to_unmask = torch.minimum(num_absorbed_nodes, step_size * torch.ones_like(num_absorbed_nodes))
        if num_nodes_to_unmask.max() < 1:
            return X

        # draw which nodes to unmask
        logits_nodes = self.gumbel.sample(X.shape[0:2]).to(self.device) - (~absorbed_nodes) * 1e9
        _, sorted_indices = logits_nodes.sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(num_nodes_to_unmask, logits_nodes.shape[1], self.device)
        nodes_to_unmask = topk.gather(-1, sorted_indices.argsort())
        assert torch.all(torch.logical_and(nodes_to_unmask, absorbed_nodes).sum(dim=-1) == num_nodes_to_unmask)

        index1, index2 = torch.where(nodes_to_unmask)

        Xclass = torch.argmax(X, dim=-1)
        new_class = torch.zeros_like(Xclass)
        w_t = torch.ones_like(X)[..., :self.node_output_dim]
        for i in range(self.node_output_dim):
            Xclass = (~nodes_to_unmask) * Xclass + nodes_to_unmask * new_class
            X_new = F.one_hot(Xclass, num_classes=self.node_output_dim + 1).float()
            d_logit = self.forward(X_new, E, y, node_mask).reshape(-1)
            d_logit = (d_logit*(self.get_temp_exponent(X, E, node_mask)))[index1]
            w_t[index1, index2, i] = torch.exp(d_logit)
            new_class = new_class + 1
        probs_x = logits_x.softmax(-1)
        guided_probs = (probs_x * w_t).reshape((-1, self.node_output_dim))
        pred_class = guided_probs.multinomial(1).reshape((X.shape[0], -1))
        Xclass = (~nodes_to_unmask) * Xclass + nodes_to_unmask * pred_class
        X = F.one_hot(Xclass, num_classes=self.node_output_dim + 1).float()
        return X

    def forward_step(self, X, E, y, node_mask, step_size=1):
        num_edges_to_unmask, num_nodes_to_unmask = self._get_num_edges_and_nodes_to_unmask(X, E, y, node_mask,
                                                                                           step_size)
        logits = self.model.kernel.model(X, E, y, node_mask)
        logits_e = logits.E
        Enew = self._unmask_edges(X, E, y, node_mask, logits_e, num_edges_to_unmask)
        num_masked_edges_diff = self._get_num_masked_edges(E) - self._get_num_masked_edges(Enew)
        assert torch.all(
            num_masked_edges_diff == num_edges_to_unmask), f"\n{num_edges_to_unmask} \n{num_masked_edges_diff}"

        logits_x = logits.X
        Xnew = self._unmask_nodes(X, E, y, node_mask, logits_x, num_nodes_to_unmask)
        num_masked_nodes_diff = self._get_num_masked_nodes(X) - self._get_num_masked_nodes(Xnew)
        assert torch.all(
            num_masked_nodes_diff == num_nodes_to_unmask), f"\n{num_nodes_to_unmask} \n{num_masked_nodes_diff}"
        new_data = general_utils.PlaceHolder(X=Xnew, E=Enew, y=y).mask(node_mask)
        return new_data

    def _generate(self, new_data, node_mask):
        assert not torch.any(~node_mask), node_mask
        num_nodes = int(new_data.X.shape[1])
        num_elements = num_nodes + num_nodes * (num_nodes - 1) / 2
        num_iters_without_guidance = int(round(num_elements - self.masked_percentage_lim * num_elements))
        for k in range(num_iters_without_guidance):
            new_data = self.model.kernel.forward_step(new_data.X, new_data.E, new_data.y, node_mask)

        while torch.any(self._get_num_masked_nodes(new_data.X) != 0) or torch.any(self._get_num_masked_edges(new_data.E) != 0):
            new_data = self.forward_step(new_data.X, new_data.E, new_data.y, node_mask, 1)

        new_data.X = new_data.X[..., :self.node_output_dim]
        assert torch.all(torch.logical_xor(torch.any(new_data.X == 1, dim=-1), ~node_mask))
        new_data.E = new_data.E[..., :self.edge_output_dim]
        assert torch.all(
            torch.sum(new_data.E == 1, dim=(-1, -2, -3)) == torch.sum(node_mask, dim=-1) ** 2 - torch.sum(node_mask,
                                                                                                          dim=-1))
                                                                                                          
        return new_data, self._num_nodes(node_mask)


class ParticleFilterDiscriminatorGuidance(DiscriminatorGuidance):
    def __init__(self, model, discriminator, config):
        super(ParticleFilterDiscriminatorGuidance, self).__init__(model, discriminator, config)
        self.num_particles = config["num_particles"]
        if self.num_particles > 0:
            self.pf_help = pf_utils.BatchedParticleFilter(self.num_particles, config["pf_resampling"], self.device)
        else:
            self.pf_help = pf_utils.UnbatchedParticleFilter(config["pf_resampling"], self.device)

        self.ess_ratio_thresh = config["ess_ratio"]
        self.debug = config["debug_pf"]
        self.pf_type = config["guidance_mode"]
        self.same_order = config["same_order"]
        self.model.kernel.same_order = self.same_order

    def get_mode(self):
        raise NotImplementedError

    def normalize(self, vec):
        return self.pf_help.normalize(vec)

    def resample(self, nu):
        assert len(nu.shape) == 1
        if self.num_particles > 0:
            nu = nu.reshape((-1, self.num_particles))
        return self.pf_help.resample(nu)

    def compute_ess(self, w):
        assert len(w.shape) == 1
        if self.num_particles > 0:
            w = w.reshape((-1, self.num_particles))
        return self.pf_help.compute_ess(w)

    def _generate(self, new_data, node_mask):
        bs = new_data.X.shape[0]
        if self.num_particles < 0:
            num_particles = new_data.X.shape[0]
        else:
            num_particles = self.num_particles
        imp_weights_prev = 1 / num_particles * torch.ones(bs, device=self.device)
        W_prev = torch.ones(bs, device=self.device)
        ess_vec = []
        num_nodes = int(new_data.X.shape[1])
        num_elements = num_nodes + num_nodes*(num_nodes - 1)/2
        num_iters_without_guidance = int(round(num_elements - self.masked_percentage_lim*num_elements))
        for k in range(num_iters_without_guidance):
            new_data = self.model.kernel.forward_step(new_data.X, new_data.E, new_data.y, node_mask)
        while torch.any(self._get_num_masked_nodes(new_data.X) != 0) or torch.any(
                self._get_num_masked_edges(new_data.E) != 0):
            new_data, node_mask, imp_weights_prev, W_prev, ess = self.forward_step(new_data.X, new_data.E, new_data.y,
                                                                                   node_mask, imp_weights_prev, W_prev)
            if self.num_particles < 0:
                ess_vec.append([float(ess)])
        new_data.X = new_data.X[..., :self.node_output_dim]
        assert torch.all(torch.logical_xor(torch.any(new_data.X == 1, dim=-1), ~node_mask))
        new_data.E = new_data.E[..., :self.edge_output_dim]
        assert torch.all(
            torch.sum(new_data.E == 1, dim=(-1, -2, -3)) == torch.sum(node_mask, dim=-1) ** 2 - torch.sum(node_mask,
                                                                                                          dim=-1))
        if self.debug:
            print("Writing ESS to file")
            file_path = os.path.join("ess", self.pf_type,
                                     f"threshold_{self.ess_ratio_thresh}_nparticles_{num_particles}_n_{new_data.X.shape[1]}.csv")
            with open(file_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(ess_vec)
        return new_data, imp_weights_prev

    def importance_sampling(self, new_data, node_mask, num_nodes, imp_weights):
        new_data = new_data.mask(node_mask)
        imp_weights = imp_weights.reshape((-1, self.num_particles))
        idx = imp_weights.multinomial(1).flatten()
        offset = self.num_particles*torch.arange(len(idx), device=self.device)
        idx = idx + offset
        X = new_data.X[idx]
        E = new_data.E[idx]
        y = new_data.y[idx]

        node_mask = node_mask[idx]
        new_data = general_utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
        num_nodes = num_nodes[idx]

        return new_data, node_mask, num_nodes

    def generate(self, batch_size):
        # For SMC, batch_size is total number of samples wanted.
        # Samples shouldn't be sampled in multiple batches outside
        # this function
        # Edit: this has been unified so other generate functions also have this behaviour
        # and that functions outside that makes call to these functions expect this behaviour
        n_vec = self.n_distr.sample((batch_size,))
        n_values, num_particles_vec = torch.unique(n_vec, return_counts=True)
        graphs = []
        st = time.time()
        max_batch_size = 2500

        particle_factor = max(1, self.num_particles)
        for n, n_samples in zip(n_values, num_particles_vec):
            while n_samples > 0:
                n_particles = min(particle_factor*n_samples, max_batch_size)
                print(f"Generating molecules with {n} nodes, using {n_particles} particle(s)")
                num_nodes, prior_data, node_mask = self.sample_p0_given_n(n_particles, n)
                new_data, imp_weights = self._generate(prior_data, node_mask)
                if self.num_particles > 0:
                    new_data, node_mask, num_nodes = self.importance_sampling(new_data, node_mask, num_nodes, imp_weights)
                graphs.extend(self._create_graph_list(new_data, num_nodes))
                assert (n_particles % particle_factor) == 0
                n_samples -= int(n_particles / particle_factor)
        print(f"Done with generation, took {time.time() - st} seconds")
        assert len(graphs) == batch_size
        return graphs


class BSDG(ParticleFilterDiscriminatorGuidance):
    def get_mode(self):
        return "bootstrap particle filter discriminator guidance"

    def forward_step(self, X, E, y, node_mask, imp_weights_prev, W_prev):
        bs = X.shape[0]
        if self.num_particles < 0:
            num_particles = X.shape[0]
        else:
            num_particles = self.num_particles
        ess = self.compute_ess(imp_weights_prev)

        # resampling
        nu = imp_weights_prev.clone()
        a_resampled = self.resample(nu)
        # if ESS > threshold, set a deterministically, and nu = 1/N
        a_deterministic = torch.arange(bs, device=self.device)

        if self.num_particles < 0:
            if ess < self.ess_ratio_thresh * num_particles:
                imp_weights_prev = 1 / num_particles * torch.ones_like(imp_weights_prev)
                a = a_resampled
            else:
                nu = 1 / num_particles * torch.ones_like(imp_weights_prev)
                a = a_deterministic
        else:
            low_ess_bool = ess < self.ess_ratio_thresh * num_particles
            low_ess_bool = low_ess_bool.repeat_interleave(num_particles)
            a = a_deterministic
            a[low_ess_bool] = a_resampled[low_ess_bool]

            imp_weights_prev[low_ess_bool] = 1/num_particles
            nu[~low_ess_bool] = 1/num_particles

        assert len(a) == bs

        W_prev = W_prev.reshape(bs)  # some stupid corner case in case of a single particle, W_prev is 0-dim
        Xa = X[a]
        Ea = E[a]
        y_a = y[a]
        node_mask_a = node_mask[a]
        W_prev_a = W_prev[a]
        imp_weights_prev_a = imp_weights_prev[a]
        nu_a = nu[a]

        # Propagate
        new_data = self.model.kernel.forward_step(Xa, Ea, y_a, node_mask_a)

        # Weight
        W_t = torch.exp((self.forward(new_data.X, new_data.E, new_data.y, node_mask_a).squeeze())*self.get_temp_exponent(new_data.X, new_data.E, node_mask_a))

        imp_weights_new = (W_t * imp_weights_prev_a / (W_prev_a * nu_a))
        imp_weights_new = self.normalize(imp_weights_new)

        return new_data, node_mask_a, imp_weights_new, W_t, ess / num_particles


class FADG(ParticleFilterDiscriminatorGuidance):
    def get_mode(self):
        return "fully adapted particle filter discriminator guidance"

    def _edges_wt(self, X, E, y, node_mask, step_size):
        bs = E.shape[0]
        n = E.shape[1]
        absorbed_edge_class, Eclass, absorbed_edges, num_absorbed_edges = self._edges_to_unmask_map(X, E)
        num_edges_to_unmask = torch.minimum(num_absorbed_edges, step_size * torch.ones_like(num_absorbed_edges))  # b,n
        # num_edges_to_unmask = torch.minimum(num_absorbed_edges, step_size)  # b,n
        w_t = torch.ones_like(E)[..., :self.edge_output_dim]
        if num_edges_to_unmask.max() < 1:
            return w_t, torch.zeros_like(Eclass).bool()
        # Determine which edges to unmask
        logits_edges = self.gumbel.sample(E.shape[0:3]).to(self.device) - (~absorbed_edges) * 1e9  # b,n,n
        logits_edges = logits_edges.reshape((bs, -1))  # b, 2*n
        _, sorted_indices = logits_edges.sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(num_edges_to_unmask, logits_edges.shape[1], self.device)
        edges_to_unmask = topk.gather(-1, sorted_indices.argsort())  # b, 2*n
        edges_to_unmask = edges_to_unmask.reshape((bs, n, n))
        if self.same_order:
            if self.num_particles > 1:
                idx = self.num_particles * torch.arange(bs // self.num_particles)
                edges_to_unmask = edges_to_unmask[idx].repeat_interleave(self.num_particles, dim=0)
            else:
                edges_to_unmask = edges_to_unmask[0].unsqueeze(0).repeat_interleave(bs, dim=0)
        assert torch.all(
            torch.logical_and(edges_to_unmask, absorbed_edges).sum(dim=(-1, -2)) == num_edges_to_unmask)

        # Replace class with predicted class for edges to be unmasked
        index1, index2, index3 = torch.where(edges_to_unmask)
        new_class = torch.zeros_like(Eclass)
        for i in range(self.edge_output_dim):
            Eclass = (~edges_to_unmask) * Eclass + edges_to_unmask * new_class
            Eclass = Eclass.triu(1)
            Eclass = Eclass + Eclass.transpose(1, 2)
            Eonehot = F.one_hot(Eclass, num_classes=absorbed_edge_class)
            E_new = general_utils.encode_no_edge(Eonehot.float())
            d_logit = self.forward(X, E_new, y, node_mask).reshape(-1)
            d_logit = (d_logit*(self.get_temp_exponent(X, E, node_mask)))[index1]
            w_t[index1, index2, index3, i] = torch.exp(d_logit)
            new_class = new_class + 1
        return w_t, edges_to_unmask

    def _nodes_wt(self, X, E, y, node_mask, step_size):
        absorbed_nodes, num_absorbed_nodes = self._nodes_to_unmask_map(X, E)
        num_nodes_to_unmask = torch.minimum(num_absorbed_nodes, step_size * torch.ones_like(num_absorbed_nodes))
        w_t = torch.ones_like(X)[..., :self.node_output_dim]
        if num_nodes_to_unmask.max() < 1:
            return w_t, torch.zeros_like(node_mask).bool()

        # draw which nodes to unmask
        logits_nodes = self.gumbel.sample(X.shape[0:2]).to(self.device) - (~absorbed_nodes) * 1e9
        _, sorted_indices = logits_nodes.sort(descending=True, dim=-1)
        topk = general_utils.batch_topk_mask(num_nodes_to_unmask, logits_nodes.shape[1], self.device)
        nodes_to_unmask = topk.gather(-1, sorted_indices.argsort())
        if self.same_order:
            if self.num_particles > 1:
                idx = self.num_particles * torch.arange(X.shape[0] // self.num_particles)
                nodes_to_unmask = nodes_to_unmask[idx].repeat_interleave(self.num_particles, dim=0)
            else:
                nodes_to_unmask = nodes_to_unmask[0].unsqueeze(0).repeat_interleave(X.shape[0], dim=0)
        assert torch.all(torch.logical_and(nodes_to_unmask, absorbed_nodes).sum(dim=-1) == num_nodes_to_unmask)

        index1, index2 = torch.where(nodes_to_unmask)

        Xclass = torch.argmax(X, dim=-1)
        new_class = torch.zeros_like(Xclass)
        for i in range(self.node_output_dim):
            Xclass = (~nodes_to_unmask) * Xclass + nodes_to_unmask * new_class
            X_new = F.one_hot(Xclass, num_classes=self.node_output_dim + 1).float()
            d_logit = self.forward(X_new, E, y, node_mask).reshape(-1)
            d_logit = (d_logit*(self.get_temp_exponent(X, E, node_mask)))[index1]
            w_t[index1, index2, i] = torch.exp(d_logit)
            new_class = new_class + 1

        return w_t, nodes_to_unmask

    def _normalize_proposal(self, logits, W_prev, W_t_nodes, W_t_edges, nodes_to_unmask, edges_to_unmask):
        probs_e = logits.E.softmax(-1)
        probs_x = logits.X.softmax(-1)
        guided_probs_e = W_t_edges * probs_e / W_prev.reshape((-1, 1, 1, 1))
        guided_probs_x = W_t_nodes * probs_x / W_prev.reshape((-1, 1, 1))

        C_t_nodes = torch.sum(guided_probs_x, dim=-1)[nodes_to_unmask]
        C_t_edges = torch.sum(guided_probs_e, dim=-1)[edges_to_unmask]
        unmask_node_bool = torch.sum(nodes_to_unmask, dim=-1).bool()
        unmask_edge_bool = torch.sum(edges_to_unmask, dim=(-1, -2)).bool()
        C_t = torch.zeros_like(W_prev)
        C_t[unmask_node_bool] = C_t_nodes
        C_t[unmask_edge_bool] = C_t_edges

        return F.normalize(guided_probs_x, dim=-1, p=1), F.normalize(guided_probs_e, dim=-1, p=1), C_t

    def pred_class(self, node_probs, edge_probs):
        n = node_probs.shape[1]

        pred_edge_class = edge_probs.reshape((-1, self.edge_output_dim)).multinomial(1).reshape((-1, n, n))
        pred_node_class = node_probs.reshape((-1, self.node_output_dim)).multinomial(1).reshape((-1, n))

        return pred_edge_class, pred_node_class

    def sample_from_proposal(self, X, E, node_probs, edge_probs, nodes_to_unmask, edges_to_unmask, W_t_nodes,
                             W_t_edges):
        pred_edge_class, pred_node_class = self.pred_class(node_probs, edge_probs)

        Xclass = torch.argmax(X, dim=-1)
        Xclass = (~nodes_to_unmask) * Xclass + nodes_to_unmask * pred_node_class
        Xnew = F.one_hot(Xclass, num_classes=self.node_output_dim + 1).float()

        Eclass = torch.argmax(E, dim=-1)
        Eclass = (~edges_to_unmask) * Eclass + edges_to_unmask * pred_edge_class
        Eclass = Eclass.triu(1)
        Eclass = Eclass + Eclass.transpose(1, 2)

        Eonehot = F.one_hot(Eclass, num_classes=self.edge_output_dim + 1)
        Enew = general_utils.encode_no_edge(Eonehot.float())

        # Extract W_t to avoid running discriminator again
        W_t_edges = W_t_edges.gather(dim=-1, index=pred_edge_class.unsqueeze(-1)).squeeze(-1)[edges_to_unmask]
        W_t_nodes = W_t_nodes.gather(dim=-1, index=pred_node_class.unsqueeze(-1)).squeeze(-1)[nodes_to_unmask]
        W_t = torch.zeros(X.shape[0], device=self.device)
        unmask_node_bool = torch.sum(nodes_to_unmask, dim=-1).bool()
        unmask_edge_bool = torch.sum(edges_to_unmask, dim=(-1, -2)).bool()
        W_t[unmask_edge_bool] = W_t_edges
        W_t[unmask_node_bool] = W_t_nodes

        return Xnew, Enew, W_t

    def forward_step(self, X, E, y, node_mask, imp_weights_prev, W_prev):
        bs = X.shape[0]
        if self.num_particles < 0:
            num_particles = X.shape[0]
        else:
            num_particles = self.num_particles

        num_edges_to_unmask, num_nodes_to_unmask = self._get_num_edges_and_nodes_to_unmask(X, E, y, node_mask,
                                                                                           1)
        if self.same_order:
            if self.num_particles > 1:
                mult = max(1, self.num_particles)
                idx = mult*torch.arange(bs // mult)
            else:
                idx = 0
            num_edges_to_unmask = num_edges_to_unmask[idx].repeat_interleave(num_particles)
            num_nodes_to_unmask = num_nodes_to_unmask[idx].repeat_interleave(num_particles)
        # Compute proposal distribution and C_t
        logits = self.model.kernel.model(X, E, y, node_mask)

        W_t_nodes, nodes_to_unmask = self._nodes_wt(X, E, y, node_mask, num_nodes_to_unmask)
        W_t_edges, edges_to_unmask = self._edges_wt(X, E, y, node_mask, num_edges_to_unmask)

        node_probs, edge_probs, C_t = self._normalize_proposal(logits, W_prev, W_t_nodes, W_t_edges, nodes_to_unmask,
                                                               edges_to_unmask)

        # compute nu_tilde
        nu_tilde = imp_weights_prev * C_t

        # then resample
        ess = self.compute_ess(imp_weights_prev)

        # resampling
        nu = self.normalize(nu_tilde)
        a_resample = self.resample(nu)
        # if ESS > threshold, set a deterministically, and nu = 1/N
        a_deterministic = torch.arange(bs, device=self.device)

        if self.num_particles < 0:
            if ess < self.ess_ratio_thresh * num_particles:
                a = a_resample
                imp_weights_prev = 1 / num_particles * torch.ones_like(imp_weights_prev)
            else:
                a = a_deterministic
                nu = 1 / num_particles * torch.ones_like(imp_weights_prev)
        else:
            low_ess_bool = ess < self.ess_ratio_thresh * num_particles
            low_ess_bool = low_ess_bool.repeat_interleave(num_particles)
            a = a_deterministic
            a[low_ess_bool] = a_resample[low_ess_bool]

            imp_weights_prev[low_ess_bool] = 1/num_particles
            nu[~low_ess_bool] = 1/num_particles

        assert len(a) == bs

        Xa = X[a]
        Ea = E[a]
        y_a = y[a]
        node_mask_a = node_mask[a]
        imp_weights_prev_a = imp_weights_prev[a]
        nu_a = nu[a]
        node_probs_a = node_probs[a]
        edge_probs_a = edge_probs[a]
        C_t_a = C_t[a]
        nodes_to_unmask_a = nodes_to_unmask[a]
        edges_to_unmask_a = edges_to_unmask[a]
        W_t_nodes_a = W_t_nodes[a]
        W_t_edges_a = W_t_edges[a]

        # Propagate
        Xnew, Enew, W_t = self.sample_from_proposal(Xa, Ea, node_probs_a, edge_probs_a, nodes_to_unmask_a,
                                                    edges_to_unmask_a, W_t_nodes_a, W_t_edges_a)

        new_data = general_utils.PlaceHolder(X=Xnew, E=Enew, y=y_a).mask(node_mask_a)

        imp_weights_new = self.normalize(imp_weights_prev_a * C_t_a / nu_a)

        return new_data, node_mask_a, imp_weights_new, W_t, ess / num_particles


guidance_dict = {"ardg": ARDG,
                 "bsdg": BSDG, "fadg": FADG}



