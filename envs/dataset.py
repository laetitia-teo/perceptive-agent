# define dataset logic
import os
import re
import os.path as op
import torch

from PIL import Image
from torch.utils.data import Dataset

class ObjectEnvBuffer(Dataset):
    """
    Buffer of all transitions in the dataset.
    Assumes everything can be loaded in memory for the specified device.
    Also assumes the number of steps is the same for every episode.
    """
    def __init__(self, datapath, gpu=True, size=(3, 64, 64)):
        
        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device

        self.datapath = datapath

        # action matrix
        self.amat = torch.Tensor(
            np.load(op.join(datapath), 'action_matrix.npy'))

        # load all data in memory
        l = os.listdir(self.datapath)
        # one dim is for epochs, another is for steps
        t = torch.zeros((0, 0) + size)

        rese = lambda s: re.search(r'^([0-9]+)_.*$', s)
        self.n_epochs = max([int(rese(s)[1]) for s in l if rese(s)])
        rese = lambda s: re.search(r'^.*_([0-9]+).*$', s)
        self.n_steps = max([int(rese(s)[1]) for s in l if rese(s)])

        for ep in range(self.n_epochs):
            tint = torch.zeros(((0,) + size))
            for step in range(self.n_steps):
                img = Image.open(op.join(self.datapath, f'{ep}_{step}.png'))
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1) 
                tint = torch.cat([tint, img.unsqueeze(0)])
            t = torch.cat([t, tint])

        self.data = t.to(self.device)
        self.amat = self.amat.to(self.device)

    def __len__(self):
        return (self.n_steps - 1) * self.n_epochs

    def __getitem__(self, idx):
        ei = idx // (self.n_steps - 1)
        si = idx % (self.n_steps - 1)
        return (self.data[ei, si:si+1], self.amat[ei, si])