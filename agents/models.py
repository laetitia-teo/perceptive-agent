# define base models for building the agent

import torch
import torch.nn.functional as F

# maybe we don't actually need this
# from torch_scatter import scatter_sum

## Utils

def mlp_fn(layer_sizes, norm=False):
    # for building MLPs easily
    layers = []
    f1 = layer_sizes[0]
    for i, f2 in enumerate(layer_sizes[1:-1]):
        layers.append(torch.nn.Linear(f1, f2))
        if norm and i == len(layer_sizes[1:-1]):
            layers.append(torch.nn.LayerNorm())
        layers.append(torch.nn.ReLU())
        f1 = f2
    layers.append(torch.nn.Linear(f1, layer_sizes[-1]))
    return torch.nn.Sequential(*layers)

## Model blocks

class PerceptionSimple(torch.nn.Module):
    """
    Simple CNN, with K output channels for K objects, similar to the one used
    in C-SWMs.
    """
    def __init__(self, K, out_ch, input_shape=(3, 64, 64)):
        super().__init__()
        self.K = K

        ch_list = ((3, 32),
                   (32, 32),
                   (32, 32),
                   (32, self.K))

        layers = []
        for ch in ch_list:
            layers.append(torch.nn.Conv2d(*ch, 3, padding=1))
            layers.append(torch.nn.BatchNorm2d(ch[-1]))
            layers.append(torch.nn.ReLU())
            # layers.append(torch.nn.MaxPool2d(2, 2))
        self.conv = torch.nn.Sequential(*layers)

        in_ch = input_shape[1] * input_shape[2]
        self.mlp = mlp_fn([in_ch, 512, 512, out_ch], norm=True)

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
    Forward step takes in states and actions.
    """
    def __init__(self, zdim, adim, h, K):
        super().__init__()

        self.zdim = zdim
        self.adim = adim
        self.h = h
        self.K = K

        self.edge_model = mlp_fn(
            [2 * self.zdim, 512, 512, self.zdim])
        self.node_model = mlp_fn(
            [(2 * self.zdim + self.adim), 512, 512, self.zdim])

        # pre-compute edge indices
        # TODO: remove self-edges
        I = torch.arange(self.K)
        ei = torch.stack(torch.meshgrid(I, I), -1).reshape((-1, 2))
        self.register_buffer('ei', ei)

    def forward(self, x, a):
        # TODO: check all this, 

        print(x.shape)
        print(a.shape)

        # assert(x.shape[1] == self.K and x.shape[2] + a.shape[-1] == self.zdim)
        
        bsize = x.shape[0]
        ei = self.ei.expand(bsize, -1, -1)

        # edges are concat of all pairs of nodes
        rx = x.reshape(-1, self.zdim)
        rei = ei.reshape(-1, 2)
        e = rx[rei]
        
        # apply edge model
        # TODO: check this
        print(e.shape)
        e = self.edge_model(e.reshape(-1, 2*self.zdim))
        e = e.reshape(bsize, self.K, self.K - 1, self.zdim)
        # aggregate
        e = e.sum(2)
        
        # apply node model
        x = torch.cat([x, a, e], -1)
        x = self.node_model(x)
        return x

## Complete model

class C_SWM(torch.nn.Module):
    """
    Contrastive Structured World Model.
    """
    def __init__(self, K, zdim, adim):
        super().__init__()
        self.K = K
        self.zdim = zdim
        # dimenson of the actions
        self.adim = adim

        self.perception = PerceptionSimple(self.K, self.zdim)
        self.transition = GNN_1(self.zdim, self.adim, self.zdim, self.K)

    def forward(self, data):
        # forward pass returns hinge loss
        # TODO: adapt for multi object
        st, st_, stp, a = data
        
        zt = self.perception(st)
        zt_ = self.perception(st_)
        ztp = self.perception(stp)

        d1 = F.mse_loss(zt + self.transition(zt, a), zt_)
        d2 = - F.relu(T.mse_loss(ztp, zt_) - GAMMA)

        return d1 + d2