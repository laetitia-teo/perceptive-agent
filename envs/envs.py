import inspect
import numpy as np
import math
import gym

from enum import IntEnum

from gym_miniworld.miniworld import MiniWorldEnv, Room
from gym_miniworld.entity import Box, COLORS, COLOR_NAMES
from gym_miniworld.params import DEFAULT_PARAMS
from gym import spaces

MIN_OBJ = 3
MAX_OBJ = 6

class MyEnv(MiniWorldEnv):
    """
    Big square room with randomly placed objects.
    For now: only boxes, and no similarity functions.
    No reward. Existential crisis for the agent.
    """
    def __init__(self,
                 min_obj=3,
                 max_obj=8,
                 size=10,
                 max_episode_steps=400,
                 **kwargs):
        assert(size > 2)
        self.size = size
        self.min_obj = min_obj
        self.max_obj = max_obj

        super().__init__(
            obs_height=64,
            obs_width=64,
            max_episode_steps=max_episode_steps,
            **kwargs)

        # allow only movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

        self.agent.cam_height = 0.5
        self.agent.cam_fov_y = 10

    def _gen_world(self):
        print('bar !')
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size)
        self.boxes = []
        self.n = np.random.randint(self.min_obj, self.max_obj + 1)
        for _ in range(self.n):
            color = np.random.choice(COLOR_NAMES)
            self.boxes.append(self.place_entity(Box(color=color)))
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, 0., done, info

class ReachBlueBox(MiniWorldEnv):
    """
    Test environment composed of one room with a random number of randomly
    colored objects.
    """
    def __init__(self, size=10, max_episode_steps=400, **kwargs):
        assert size >= 2
        self.size = size
        self.distractors = np.random.randint(MIN_OBJ, MAX_OBJ + 1)
        self.target_color = COLOR_NAMES.pop(0)

        super().__init__(
            max_episode_steps=max_episode_steps,
            **kwargs)

        # allow only movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

        # number of objects

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size)
        self.boxes = []
        for _ in range(self.distractors):
            color = np.random.choice(COLOR_NAMES)
            self.boxes.append(self.place_entity(Box(color=color)))
        self.target_box = self.place_entity(Box(color=self.target_color))
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.target_box):
            reward += self._reward()
            done = True

        return obs, reward, done, info

class ObjectEnv(MiniWorldEnv):
    """
    An environment with one room and several objects.
    One of those objects can be moved forward and turned.
    The agent (camera) is on a top corner of the room.
    """
    # class Actions():
    #     """
    #     Describing the possible actions.
    #     """
    #     def __init__():
    #           

    def __init__(self):
        self.size = 6
        # only the blue box can be moved
        self.n_boxes = 4

        self.cam_height = 4.
        self.cam_pitch = -50
        self.cam_fov_y = 70

        super().__init__(
            obs_height=64,
            obs_width=64,)
        self.n_actions = self.actions.move_back + 1
        self.action_space = spaces.Discrete(
            (self.n_actions) * self.n_boxes)

    def action_to_one_hot(self, action):
        """
        Converts action to a set of one-hot vectors, one for each object.
        """
        abs_action = action % self.n_actions
        box_idx = action // self.n_actions
        action_mat = np.zeros((self.n_boxes, self.n_actions))

        action_mat[box_idx, abs_action] = 1.
        return np.array([action_mat])

    def reset(self):
        super().reset()
        # change camera attributes
        self.agent.cam_height = self.cam_height
        self.agent.cam_pitch = self.cam_pitch
        self.agent.cam_fov_y = self.cam_fov_y
        self.agent.radius = 0.
        # maybe change fov
        return self.render_obs()

    # relative, angular actions

    def move_box(self, step, drift):
        
        next_pos = (
            self.target_box.pos + 
            self.target_box.dir_vec * step)

        if self.intersect(self.target_box, next_pos, self.target_box.radius-0.4):
            return False

        # check if box does not move too close to camera
        if next_pos[0] + next_pos[2] <= 2.5:
            return False

        self.target_box.pos = next_pos

        return True

    def turn_box(self, turn_angle):
        
        turn_angle *= (math.pi / 180)
        self.target_box.dir += turn_angle

        return True

    # absolute, cartesian actions

    def move_box_abs(self, box_idx, step, dir_vec):
        target_box = self.boxes[box_idx]

        next_pos = (
            target_box.pos + 
            dir_vec * step)

        if self.intersect(target_box, next_pos, target_box.radius-0.4):
            return False

        # check if box does not leave room
        if abs(self.size/2 - next_pos[0]) >= self.size/2:
            return False
        if abs(self.size/2 - next_pos[2]) >= self.size/2:
            return False

        target_box.pos = next_pos

        return True

    def move_box_left(self, box_idx, step):
        dir_vec = np.array([1., 0., 0.])
        return self.move_box_abs(box_idx, step, dir_vec)
        
    def move_box_right(self, box_idx, step):
        dir_vec = np.array([-1., 0., 0.])
        return self.move_box_abs(box_idx, step, dir_vec)
        
    def move_box_front(self, box_idx, step):
        dir_vec = np.array([0., 0., 1.])
        return self.move_box_abs(box_idx, step, dir_vec)
        
    def move_box_back(self, box_idx, step):
        dir_vec = np.array([0., 0., -1.])
        return self.move_box_abs(box_idx, step, dir_vec)
        

    def step(self, action, time_limit=False):
        """
        Perform one action and update the simulation.
        """
        self.step_count += 1

        # no domain randomization
        fwd_step = self.params.sample(None, 'forward_step') * 3

        abs_action = action % self.n_actions
        box_idx = action // self.n_actions

        if abs_action == self.actions.move_forward:
            self.move_box_left(box_idx, fwd_step)
        elif abs_action == self.actions.move_back:
            self.move_box_right(box_idx, fwd_step)
        elif abs_action == self.actions.turn_left:
            self.move_box_front(box_idx, fwd_step)
        elif abs_action == self.actions.turn_right:
            self.move_box_back(box_idx, fwd_step)

        # generate observation
        obs = self.render_obs()

        # check for time limit
        if self.step_count >= self.max_episode_steps and time_limit:
            done = True
            reward = 0
            return obs, reward, done, {}

        reward = 0
        done = False

        return obs, reward, done, {}

    def _gen_world(self):
        color_names = list(COLOR_NAMES)
        color_names.pop(2) # remove grey
        # we can change the textures in the kwargs here
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_height=4.)
        
        # place agent
        self.place_entity(
            ent=self.agent,
            pos=np.array([0.1, 0., 0.1]),
            dir=-math.pi/4)
        
        self.boxes = []
        for i in range(self.n_boxes):
            self.boxes.append(
                self.place_entity(Box(color=color_names[i])))

###### Utils

def register_envs():
    """
    Register all envs.
    """
    module_name = __name__
    global_vars = globals()

    # Iterate through global names
    for global_name in sorted(list(global_vars.keys())):
        env_class = global_vars[global_name]

        if not inspect.isclass(env_class):
            continue

        if not issubclass(env_class, gym.core.Env):
            continue

        if env_class is MiniWorldEnv:
            continue

        # Register the environment with OpenAI Gym
        gym_id = 'MiniWorld-%s-v0' % (global_name)
        entry_point = '%s:%s' % (module_name, global_name)

        try:
            gym.envs.registration.register(
                id=gym_id,
                entry_point=entry_point,)
            print(gym_id)
        except gym.error.Error:
            pass

# register_envs()

# def reregister(name='MultiObj'):
#     """
#     Delete env if it's registered and re-register it, simply register it if not
#     already registered.
#     """
#     env_id = f'MiniWorld-{name}-v0'
#     if env_id in gym.envs.registry.env_specs:
#         del gym.envs.registry.env_specs[env_id]

#     gym.register(
#         id=env_id,
#         entry_point=MultiObj)