import torch
import os
import torch.nn.functional as F
from ..transformer import general_utils


from torch_geometric.data import InMemoryDataset


def create_graph_list(data, num_nodes):
    Xclass = torch.argmax(data.X, dim=-1)
    Eclass = torch.argmax(data.E, dim=-1)

    graph_list = []
    for i, n in enumerate(num_nodes):
        node_features = Xclass[i, :n]
        edge_features = Eclass[i, :n, :n]
        graph_list.append((node_features, edge_features))
    return graph_list


class GeneratedDataset(InMemoryDataset):
    def __init__(self, config):
        self.root_dir = config["gen_data_path"]
        self.device = config["device"]
        self.name = config["task"]
        super(GeneratedDataset, self).__init__(self.root_dir)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        raise NotImplementedError("Cannot download data, it must be generated")

    @property
    def raw_file_names(self):
        return ["gen_dataset.pt"]

    def process(self):
        # Read data into huge `Data` list.
        if str(self.device) == "cpu":
            data_list = torch.load(os.path.join(self.root_dir, "raw", "gen_dataset.pt"),
                                   map_location=torch.device("cpu"))
        else:
            data_list = torch.load(os.path.join(self.root_dir, "raw", "gen_dataset.pt"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    def make_list(self):
        graph_list = []
        for i in range(len(self)):
            data = self.get(i).to(self.device)
            try:
                dense_data, node_mask = general_utils.to_dense(data.x, data.edge_index, data.edge_attr, torch.zeros(len(data.x), device=self.device).long())
            except RuntimeError as e:
                if data.edge_index.size(1) == 0:
                    X = data.x.unsqueeze(0)
                    num_nodes = data.x.shape[0]
                    E = torch.zeros((1, num_nodes, num_nodes), device=self.device).long()
                    E = F.one_hot(E, num_classes=1)
                    node_mask = torch.ones((1, num_nodes), device=self.device).bool()
                    dense_data = general_utils.PlaceHolder(X=X.long(), E=E.long(), y=None)
                else:
                    print(len(data.edge_index))
                    print(data.x)
                    print(data.edge_index)
                    raise e
            dense_data = dense_data.mask(node_mask)
            graph_list.extend(create_graph_list(dense_data, torch.sum(node_mask, dim=1)))
        return graph_list

