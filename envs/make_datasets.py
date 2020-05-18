# makes datasets of images by performing random actions in environments
import os.path as op
import pathlib
import gym
import numpy as np

from PIL import Image
from envs import ObjectEnv

# sys.path.append(..)

NEPISODES = 5000
NSTEPS = 10
DATAPATH = op.join('..', 'data', 'dset1')

envname = 'MiniWorld-ObjectEnv-v0'

# unregister and register env
if envname in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[envname]

gym.register(
    id=envname,
    entry_point=ObjectEnv)

env = gym.make(envname)

def save_obs(obs, path, ep, idx):
    im = Image.fromarray(obs)
    im.save(op.join(path, f"{ep}_{idx}.png"))

pathlib.Path(DATAPATH).mkdir(parents=True, exist_ok=True)

mat = np.zeros((NEPISODES, NSTEPS, env.n_boxes, env.n_actions))

for ep in range(NEPISODES):
    print(f'Episode {ep}; rollout for {NSTEPS} random steps')
    obs = env.reset()
    save_obs(obs, DATAPATH, ep, 0)

    for i in range(NSTEPS):
        action = env.action_space.sample()
        action_mat = env.action_to_one_hot(action)
        obs, _, _, _ = env.step(action)

        save_obs(obs, DATAPATH, ep, i+1)

        mat[ep, i] = action_mat

np.save(op.join(DATAPATH, 'action_matrix.npy'), mat)

print('done')