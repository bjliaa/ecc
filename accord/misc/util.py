import numpy as np
import pandas as pd
import os


def discounted(rews, gamma):
    return np.sum([rews[i] * (gamma**i) for i in range(len(rews))])


def rclip(rew):
    return np.sign(rew)


def datasave(datadict, env_str, steps, wid, **kwargs):
    dir_str = f"data/{env_str}/steps{steps}/"
    os.makedirs(dir_str, exist_ok=True)
    file_str = dir_str + "model-id-" + str(wid) + ".csv"
    df = pd.DataFrame(datadict)
    for key in kwargs:
        df[key] = kwargs[key]
    df.to_csv(file_str)


def modelsave(mdl, env_str, step, wid):
    mdl.save(modelfilestr(env_str, step, wid))


def modelfilestr(env_str, step, wid):
    dir_str = f"models/{env_str}/step{step}/"
    os.makedirs(dir_str, exist_ok=True)
    file_str = dir_str + "model-id-" + f"{wid}" + ".h5"
    return file_str


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
