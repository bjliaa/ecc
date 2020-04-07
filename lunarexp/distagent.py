import tensorflow as tf
from models import Probnet


class DistAgent(tf.Module):
    def __init__(self,
                 action_len,
                 dense=32,
                 supportsize=29,
                 vmin=-7.0,
                 vmax=7.0,
                 starteps=1.0,
                 lr=1e-4,
                 adameps=1.5e-4,
                 name="distagent"):
        super(DistAgent, self).__init__(name=name)

        self.action_len = action_len
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                                  epsilon=adameps)
        self.losses = tf.keras.losses.KLDivergence(
            reduction=tf.keras.losses.Reduction.NONE)
        self.kldloss = tf.keras.losses.KLDivergence()

        with tf.name_scope("probnet"):
            self.probnet = Probnet(action_len, dense, supportsize)
        with tf.name_scope("target_probnet"):
            self.targetnet = Probnet(action_len, dense, supportsize)

        self.supp = tf.constant(tf.linspace(vmin, vmax, supportsize),
                                shape=(supportsize, 1))
        self.dz = tf.constant((vmax - vmin) / (supportsize - 1))
        self.vmin = tf.constant(vmin)
        self.vmax = tf.constant(vmax)
        self.supportsize = tf.constant(supportsize, dtype=tf.int32)
        self.eps = tf.Variable(starteps, trainable=False, name="epsilon")

    @tf.function
    def eps_greedy_action(self, state, epsval):
        self.eps.assign(epsval)
        dice = (tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32) <
                self.eps)
        raction = tf.random.uniform([1],
                                    minval=0,
                                    maxval=self.action_len,
                                    dtype=tf.int64)
        qaction = tf.argmax(self.qvalues(state))
        return tf.where(dice, raction, qaction)

    @tf.function
    def probvalues(self, states):
        return tf.squeeze(self.probnet(states))

    @tf.function
    def qvalues(self, states):
        ds = self.probnet(states)
        return tf.squeeze(tf.matmul(ds, self.supp))

    @tf.function
    def t_probvalues(self, states):
        return tf.squeeze(self.targetnet(states))

    @tf.function
    def t_qvalues(self, states):
        ds = self.targetnet(states)
        return tf.squeeze(tf.matmul(ds, self.supp))

    @tf.function
    def update_target(self):
        q_vars = self.probnet.trainable_variables
        t_vars = self.targetnet.trainable_variables
        for var_q, var_t in zip(q_vars, t_vars):
            var_t.assign(var_q)

    @tf.function
    def train(self, states, actions, drews, gexps, endstates, dones):
        with tf.GradientTape() as tape:
            batch_size = tf.shape(states)[0]
            brange = tf.range(0, batch_size)
            indices = tf.stack([brange, actions], axis=1)
            chosen_dists = tf.gather_nd(self.probvalues(states), indices)

            end_actions = tf.cast(tf.argmax(self.t_qvalues(endstates), axis=1),
                                  dtype=tf.int32)

            indices = tf.stack([brange, end_actions], axis=1)
            chosen_end_dists = tf.gather_nd(self.t_probvalues(endstates),
                                            indices)

            dmask = (1.0 - dones) * gexps
            Tzs = tf.clip_by_value(drews + dmask * self.supp, self.vmin,
                                   self.vmax)
            Tzs = tf.transpose(Tzs)
            bs = (Tzs - self.vmin) / self.dz

            ls = tf.cast(tf.floor(bs), tf.int32)
            us = tf.cast(tf.math.ceil(bs), tf.int32)
            condl = tf.cast(
                tf.cast((us > 0), tf.float32) * tf.cast(
                    (us == ls), tf.float32), tf.bool)
            condu = tf.cast(
                tf.cast((ls < self.supportsize - 1), tf.float32) * tf.cast(
                    (us == ls), tf.float32), tf.bool)
            ls = tf.where(condl, ls - 1, ls)
            us = tf.where(condu, us + 1, us)

            luprob = (tf.cast(us, tf.float32) - bs) * chosen_end_dists
            lshot = tf.one_hot(ls, self.supportsize)
            ml = tf.einsum('aj,ajk->ak', luprob, lshot)
            ulprob = (bs - tf.cast(ls, tf.float32)) * chosen_end_dists
            ushot = tf.one_hot(us, self.supportsize)
            mu = tf.einsum('aj,ajk->ak', ulprob, ushot)

            target = ml + mu
            losses = self.losses(target, chosen_dists)

            # Kullback–Leibler divergence
            loss = self.kldloss(tf.stop_gradient(target), chosen_dists)

        gradients = tape.gradient(loss, self.probnet.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.optimizer.apply_gradients(
            zip(gradients, self.probnet.trainable_variables))
        return losses

    def save(self, filestr):
        self.probnet.save_weights(filestr)

    def load(self, filestr):
        self.probnet.load_weights(filestr)
