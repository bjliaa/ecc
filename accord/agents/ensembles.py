import tensorflow as tf
import numpy as np
from accord.agents.models import Probnet
from accord.agents.models import Ampnet
from accord.agents.distributional import DistAgent
from accord.agents.ampdist import AmpDistAgent


class JointEnsemble(tf.keras.Model):
    def __init__(self,
                 action_len,
                 size=3,
                 inputshape=(84, 84, 4),
                 starteps=0.001,
                 supportsize=51,
                 vmin=-10,
                 vmax=10,
                 name="AvgJoint"):
        super(JointEnsemble, self).__init__(name=name)

        self.agents = [
            DistAgent(action_len, name=f"A{i}") for i in range(size)
        ]

        self.supportsize = np.int32(supportsize)
        self.vmin = np.float32(vmin)
        self.vmax = np.float32(vmax)
        self.eps = tf.Variable(starteps, trainable=False, name="epsilon")
        self.action_len = action_len
        self.size = size
        self.supp = tf.expand_dims(tf.linspace(self.vmin, self.vmax,
                                               self.supportsize),
                                   axis=1)

    @tf.function
    def call(self, x, training=False):
        out = [self.agents[i].probnet(x) for i in range(self.size)]
        out = tf.reduce_mean(out, axis=0)
        return tf.squeeze(out)

    @tf.function
    def avgQvals(self, x):
        ds = self.call(x)
        return tf.squeeze(tf.matmul(ds, self.supp))

    @tf.function
    def avgQ_action(self, state, epsval):
        self.eps.assign(epsval)
        dice = (tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32) <
                self.eps)
        raction = tf.random.uniform([1],
                                    minval=0,
                                    maxval=self.action_len,
                                    dtype=tf.int64)
        qaction = tf.argmax(self.avgQvals(state))
        return tf.where(dice, raction, qaction)

    def load(self, files):
        assert len(files) == len(self.agents)
        for i in range(len(files)):
            self.agents[i].load(files[i])


class AmpJointEnsemble(tf.keras.Model):
    def __init__(self,
                 action_len,
                 size=3,
                 inputshape=(84, 84, 4),
                 starteps=0.001,
                 supportsize=51,
                 vmin=-10,
                 vmax=10,
                 name="AvgJoint"):
        super(AmpJointEnsemble, self).__init__(name=name)

        self.inp = tf.keras.Input(inputshape, dtype=tf.float32)
        self.nets = [
            Probnet(action_len, supportsize=supportsize, name=f"Net{i}")
            for i in range(size)
        ]
        self.netsout = [self.nets[i](self.inp) for i in range(size)]

        self.sp = tf.keras.Model(inputs=self.inp, outputs=self.netsout)
        self.supportsize = np.int32(supportsize)
        self.vmin = np.float32(vmin)
        self.vmax = np.float32(vmax)
        self.eps = tf.Variable(starteps, trainable=False, name="epsilon")
        self.action_len = action_len
        self.size = size

    @tf.function
    def call(self, x, training=False):
        return self.sp(x)

    @tf.function
    def avgQvals(self, x):
        supp = tf.expand_dims(tf.linspace(self.vmin, self.vmax,
                                          self.supportsize),
                              axis=1)
        avgqs = tf.reduce_mean(tf.matmul(self.sp(x), supp), axis=0)
        return tf.squeeze(avgqs)

    @tf.function
    def avgQ_action(self, state, epsval):
        self.eps.assign(epsval)
        dice = (tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32) <
                self.eps)
        raction = tf.random.uniform([1],
                                    minval=0,
                                    maxval=self.action_len,
                                    dtype=tf.int64)
        qaction = tf.argmax(self.avgQvals(state))
        return tf.where(dice, raction, qaction)

    def load(self, ws):
        for i in range(self.size):
            self.nets[i].load_weights(ws[i])
