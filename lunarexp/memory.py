import numpy as np
from random import choices, randint
from math import floor


class ReplayBuffer(object):
    def __init__(self, size, batchsize):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.batchsize = batchsize
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        states, actions, drews, gexps, endstates, dones = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in idxes:
            xp = self._storage[i]
            ss, ac, dr, ge, es, dn = xp
            states.append(np.array(ss, copy=False))
            actions.append(np.array(ac, copy=False))
            drews.append(dr)
            gexps.append(ge)
            endstates.append(np.array(es, copy=False))
            dones.append(dn)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(drews, dtype=np.float32),
            np.array(gexps, dtype=np.float32),
            np.array(endstates, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def sample(self):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [
            randint(0,
                    len(self._storage) - 1) for _ in range(self.batchsize)
        ]
        return self._encode_sample(idxes)
