# file for training the agent/transition model

import os.path as op
import sys
import torch
import torch.nn.functional as F

sys.path.append(op.join('..', 'envs'))

from pathlib import Path
from torch.utils.data import DataLoader
from dataset import ObjectEnvBuffer
from models import C_SWM

RUNIDX = 0
DATAPATH = op.join('..', 'data', 'dset1')
SAVEPATH = op.join('..', 'data', 'saves', f'run{RUNIDX}')

BSIZE = 128
NEPOCHS = 10
LR = 1e-4

GAMMA = 1.
K = 4
ZDIM = 2


def train(cswm, dl, n_epochs):
    opt = torch.optim.Adam(cswm.parameters(), lr=LR)
    for ep in range(n_epochs):
        for i, data in enumerate(dl):
            loss = cswm(data)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f'Epoch {ep}, batch {i}, loss {loss.item()}')
        print('saving model')
        torch.save(cswm.state_dict(), op.join(SAVEPATH, f'epoch_{ep}.pt'))

oeb = ObjectEnvBuffer(DATAPATH)
dl = DataLoader(oeb, batch_size=BSIZE, shuffle=True)

cswm = C_SWM(K, ZDIM)
cswm = cswm.to(oeb.device)

data = next(iter(dl))

if __name__ == "__main__":
    oeb = ObjectEnvBuffer(DATAPATH)
    dl = DataLoader(oeb, batch_size=BSIZE, shuffle=True)

    cswm = C_SWM(K, ZDIM)
    cswm.to(oeb.device)
    train(C_SWM, dl, NEPOCHS)