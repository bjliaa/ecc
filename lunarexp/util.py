import gym
import numpy as np
from collections import deque
from gym import spaces


def scale(rew):
    return rew / 100


class Linear:
    def __init__(self, startval, endval, exploresteps):
        self.exp = exploresteps
        self.sval = startval
        self.endval = endval
        self.dydx = (endval - startval) / exploresteps

    def __call__(self, t):
        if t <= self.exp:
            return self.sval + t * self.dydx
        else:
            return self.endval


class RewMonitor(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.eprew = 0.0

    def reset(self, **kwargs):
        self.eprew = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.eprew += rew
        if done:
            info["Game Over"] = True
            info["Episode Score"] = self.eprew
        else:
            info["Game Over"] = False
        return obs, rew, done, info


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action, render=False):
        total_reward = 0.0
        done = None
        obs = []
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if render:
                self.env.render()
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class StackEnv(gym.Wrapper):
    def __init__(self, env, n_frames):
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return list(self.frames)
