import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_cross_entropy_per_element(mask, logits, target):
    return F.cross_entropy(logits[mask], target[mask], reduction='none')


def uniform_weighted_cross_entropy_loss(mask, logits, target):
    target = target.long()
    n_masks = torch.sum(mask, dim=1)
    weights = 1 / n_masks
    weights = weights.repeat_interleave(n_masks)

    out = masked_cross_entropy_per_element(mask, logits, target)
    out = weights*out
    out = out / torch.sum(weights)
    return torch.sum(out)


def graph_uniform_cross_entropy_loss(node_dict, edge_dict):
    node_mask = node_dict["mask"]  # mask indicates which elements that have been predicted
    edge_mask = edge_dict["mask"]

    node_ce_per_element = masked_cross_entropy_per_element(node_mask, node_dict["logits"], node_dict["target"])
    edge_ce_per_element = masked_cross_entropy_per_element(edge_mask, edge_dict["logits"], edge_dict["target"])

    n_node_masks = torch.sum(node_mask, dim=1)
    n_edge_masks = torch.sum(edge_mask, dim=1)
    n_masks = n_node_masks + n_edge_masks
    batch_size = n_masks.shape[0]
    weights = node_dict["D"] / n_masks  # Treating nodes and edges equal, weight hence D/n_masked_elements (element=node or edge)

    node_weights = node_dict["lambda"]*weights.repeat_interleave(n_node_masks)
    edge_weights = edge_dict["lambda"]*weights.repeat_interleave(n_edge_masks)

    node_loss = torch.sum(node_weights*node_ce_per_element) / batch_size
    edge_loss = torch.sum(edge_weights*edge_ce_per_element) / batch_size

    return node_loss, edge_loss


class UniformWeightedCrossEntropyLoss(nn.Module):
    def forward(self, out, target):
        return uniform_weighted_cross_entropy_loss(out['mask'], out['logits'], target)


class LogitWeightedCrossEntropyLoss(nn.Module):
    def forward(self, out, target):
        target = target.long()
        mask = out['mask']
        logits = out['logits'][:, :, :-1]
        add_logits = out['logits'][:,:, -1]

        n_masks = torch.sum(mask, dim=1)
        weights = 1 / n_masks
        weights = weights.repeat_interleave(n_masks)

        add_prob = -1e9*torch.ones_like(mask)
        add_prob[mask] = add_logits[mask]
        log_add_prob = F.log_softmax(add_prob, dim=1)

        out = F.cross_entropy(logits[mask], target[mask], reduction='none')
        out = weights*(out - log_add_prob[mask])
        out = out / torch.sum(weights)
        return torch.sum(out)


