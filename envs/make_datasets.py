# makes datasets of images by performing random actions in environments
import os.path as op
import pathlib
import gym

from PIL import Image
from envs import ObjectEnv

# sys.path.append(..)

NEPISODES = 5000
NDATAPOINTS = 10
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


for ep in range(NEPISODES):
    print(f'Episode {ep}; rollout for {NDATAPOINTS} random steps')
    obs = env.reset()
    save_obs(obs, DATAPATH, 0, 0)
    for i in range(NDATAPOINTS):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)

        save_obs(obs, DATAPATH, ep, i+1)

print('done')