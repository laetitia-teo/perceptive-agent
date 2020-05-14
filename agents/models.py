# define base models for building the agent

import torch
import torch.nn.functional as F

# maybe we don't actually need this
from torch_scatter import scatter_sum

def mlp_fn(layer_sizes):
    layers = []
    f1 = layer_sizes[0]
    for i, f2 in enumerate(layer_sizes[1:-1]):
        layers.append(Linear(f1, f2))
        layers.append(ReLU())
        f1 = f2
    layers.append(Linear(f1, layer_sizes[-1]))
    return Sequential(*layers)

class PerceptionSimple(torch.nn.Module):
    """
    Simple CNN, with K output channels for K objects, similar to the one used
    in C-SWMs.
    """
    def __init__(self, K, input_shape=(3, 64, 64)):
        super().__init__()
        self.K = K

        ch_list = ((3, 32),
                   (32, 32),
                   (32, 32),
                   (32, self.K))

        layers = []
        for ch in ch_list:
            layers.append(torch.nn.Conv2d(*ch, 3, padding=1))
            layers.append(torch.nn.BatchNorm2d())
            layers.append(torch.nn.ReLU())
            # layers.append(torch.nn.MaxPool2d(2, 2))
        self.conv = torch.nn.Sequential(*layers)

        in_ch = input_shape[1] * input_shape[2]
        self.mlp = mlp_fn([in_ch, 512, 512, 512])

    def forward(self, img):
        bsize = img.shape[0]
        fmap = self.conv(img)
        # object vectors
        vecs = fmap.view((bsize, self.K, -1))
        return self.mlp(vecs)

class GNN_1(torch.nn.Module):
    """
    This implem expects complete graphs of constant node number K across the
    batch.
    One single message-passing step.
    """
    def __init__(self, in_ch, out_ch, K):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.K = K

        self.edge_model = mlp_fn([2 * self.in_ch, 512, 512, self.in_ch])
        self.node_model = mlp_fn([self.in_ch, 512, 512, self.in_ch])
        self.mlp = mlp_fn([512, 512, 512, self.out_ch])

        # pre-compute edge indices
        I = torch.arange(self.K)
        ei = torch.stack(torch.meshgrid(I, I), -1).view((-1, 2))
        self.register_buffer('ei', ei)

    def forward(self, x):
        # TODO: check all this, 

        assert(x.shape[1] == self.K and x.shape[2] == self.in_ch)
        
        bsize = x.shape[0]
        ei = self.ei.expand(bsize, -1, -1)
        
        # edges are concat of all pairs of nodes
        rx = x.reshape(-1, self.in_ch)
        rei = ei.reshape(-1, 2)
        e = rx[rei]
        
        # apply edge model
        e = self.edge_model(e).reshape(bsize, self.K, self.K, self.in_ch)
        # aggregate
        e = e.sum(2)
        
        # apply node model
        x = x + e
        x = self.node_model(x) # skip connection ?
        return x