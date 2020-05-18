# define dataset logic
import os
import re
import os.path as op
import numpy as np
import torch

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DATAPATH = op.join('..', 'data', 'dset1')

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
            np.load(op.join(datapath, 'action_matrix.npy')))

        # load all data in memory
        l = os.listdir(self.datapath)
        # one dim is for epochs, another is for steps

        rese = lambda s: re.search(r'^([0-9]+)_.*$', s)
        self.n_epochs = max([int(rese(s)[1]) for s in l if rese(s)])
        rese = lambda s: re.search(r'^.*_([0-9]+).*$', s)
        self.n_steps = max([int(rese(s)[1]) for s in l if rese(s)])
        
        t = torch.zeros((self.n_epochs, self.n_steps + 1) + size)

        print('loading dataset...')
        # for ep in tqdm(range(self.n_epochs)):
        for ep in tqdm(range(200)):
            for step in range(self.n_steps + 1):
                img = Image.open(op.join(self.datapath, f'{ep}_{step}.png'))
                img = np.array(img).astype(np.float32)
                img = torch.from_numpy(img).permute(2, 0, 1) 
                t[ep, step] = img
        print('done')

        self.data = t.to(self.device)
        self.amat = self.amat.to(self.device)

    def __len__(self):
        return (self.n_steps) * self.n_epochs

    def __getitem__(self, idx):
        ei = idx // self.n_steps
        si = idx % self.n_steps
        st = self.data[ei, si]
        st_ = self.data[ei, si+1]

        rei = np.random.randint(self.n_epochs)
        rsi = np.random.randint(self.n_steps)
        stp = self.data[rei, rsi]

        return (st, st_, stp, self.amat[ei, si])