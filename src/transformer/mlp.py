import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


activation_functions = {
    'relu': nn.ReLU,
    'leaky': nn.LeakyReLU,
    'swish': Swish
}


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden_layers, bn=False, activation_function='relu', hidden_dim=128):
        super(MultiLayerPerceptron, self).__init__()
        if n_hidden_layers > 0:
            module_list = [nn.Linear(in_dim, hidden_dim)]
            if bn:
                module_list.append(nn.BatchNorm1d(hidden_dim))
            module_list.append(activation_functions[activation_function]())
            for _ in range(1, n_hidden_layers):
                module_list.append(nn.Linear(hidden_dim, hidden_dim))
                if bn:
                    module_list.append(nn.BatchNorm1d(hidden_dim))
                module_list.append(activation_functions[activation_function]())
            module_list.append(nn.Linear(hidden_dim, out_dim))
        else:
            module_list = [nn.Linear(in_dim, out_dim)]
        self.mlp = nn.Sequential(*module_list)

    def forward(self, x):
        return self.mlp(x)
