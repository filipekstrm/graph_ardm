import torch
import torch_geometric as ptg
import torch.nn.functional as F


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = ptg.utils.to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = ptg.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = ptg.utils.to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def from_dense_to_sparse(X, E, node_mask, device):
    Eclass = E.argmax(dim=-1)
    edge_index, edge_attr = ptg.utils.dense_to_sparse(Eclass)

    # Consider the case when there is padding
    num_padded = torch.sum(~node_mask, dim=-1).roll(1, 0)
    num_padded[0] = 0
    offset = num_padded.cumsum(0)
    num_edges = torch.sum(Eclass > 0, dim=(1, 2))
    offset = offset.repeat_interleave(num_edges)
    edge_index = edge_index - offset.unsqueeze(0)

    # Edge attr to onehot
    edge_attr = F.one_hot(edge_attr, num_classes=E.shape[-1]).float()

    x = X[node_mask]
    num_nodes = torch.sum(node_mask, dim=-1)
    batch = torch.arange(X.shape[0], device=device).repeat_interleave(num_nodes)
    return x, edge_index, edge_attr, batch


def batch_topk_mask(k, max_size, device):
    """

    :param k: Vector with number of elements to keep
    :param max_size: Integer specifying size of second dimension
    :return: A tensor of size len(k)xmax_size, where for row i, the first k[i] elements will be True and False otherwise
    """
    if len(k) > 1:
        k = k.squeeze()
    assert len(k.shape) == 1, "k vector has too many dimensions"
    assert k.max() <= max_size, "max_size is smaller than some of the value(s) in k"
    bs = k.shape[0]
    topk = torch.arange(max_size, device=device).repeat((bs, 1))
    topk = topk < k.unsqueeze(-1)
    return topk


def get_time_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor([10000], device=timesteps.device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device)*(-emb))
    emb = timesteps.unsqueeze(-1) * emb.unsqueeze(0)
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def __str__(self):
        if torch.is_tensor(self.X):
            x_string = f"X={self.X.shape}"
        else:
            x_string = f"{type(self.X)}"

        if torch.is_tensor(self.E):
            e_string = f"E={self.E.shape}"
        else:
            e_string = f"E={type(self.E)}"

        if torch.is_tensor(self.y):
            y_string = f"y={self.y.shape}"
        else:
            y_string = f"y={type(self.y)}"

        return f"PlaceHolder object with {x_string}, {e_string}, {y_string}"