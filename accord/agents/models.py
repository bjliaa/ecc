import tensorflow as tf


class Ampnet(tf.keras.Model):
    def __init__(self,
                 action_len,
                 vsize,
                 dense=512,
                 supportsize=51,
                 name="ampnet"):
        super(Ampnet, self).__init__(name=name)
        self.action_len = action_len
        self.vsize = vsize
        self.probnets = [
            Probnet(action_len, dense=dense, name=f"Net{i}")
            for i in range(vsize)
        ]
        self.supp = tf.constant(tf.linspace(-10.0, 10.0, supportsize),
                                shape=(supportsize, 1))

    @tf.function
    def call(self, x, training=False):
        out = [self.probnets[i](x) for i in range(self.vsize)]
        out = tf.reduce_mean(out, axis=0)
        return tf.squeeze(out)

    # @tf.function
    def update(self, wlst):
        for i in range(self.vsize):
            q_vars = wlst[i]
            t_vars = self.probnets[i].trainable_variables
            for var_q, var_t in zip(q_vars, t_vars):
                var_t.assign(var_q)

    @tf.function
    def qvalues(self, states):
        ds = self.call(states)
        return tf.squeeze(tf.matmul(ds, self.supp))

    @tf.function
    def epsaction(self, state, epsval):
        dice = (tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32) <
                epsval)
        raction = tf.random.uniform([1],
                                    minval=0,
                                    maxval=self.action_len,
                                    dtype=tf.int64)
        qaction = tf.argmax(self.qvalues(state))
        return tf.where(dice, raction, qaction)


class Probnet(tf.keras.Model):
    def __init__(self, action_len, dense=512, supportsize=51, name="probnet"):
        super(Probnet, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            kernel_size=8,
            strides=4,
            activation="relu",
            input_shape=(84, 84, 4),
        )
        self.conv2 = tf.keras.layers.Conv2D(64,
                                            kernel_size=4,
                                            strides=2,
                                            activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(64,
                                            kernel_size=3,
                                            strides=1,
                                            activation="relu")
        self.flat = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(dense, activation="relu")
        self.probs = ProbOutput(input_shape=(dense),
                                outdim=action_len,
                                gridsize=supportsize)

    @tf.function
    def call(self, x, training=False):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.flat(y)
        y = self.dense(y)
        probout = self.probs(y)
        return probout


class ProbOutput(tf.keras.Model):
    def __init__(self, input_shape, outdim, gridsize=51):
        super(ProbOutput, self).__init__()
        self.inp = tf.keras.Input(input_shape)
        self.outp = [
            tf.keras.layers.Dense(gridsize, activation="softmax")(self.inp)
            for _ in range(outdim)
        ]
        self.model = tf.keras.Model(inputs=self.inp, outputs=self.outp)

    @tf.function
    def call(self, x, training=False):
        return tf.stack(self.model(x), axis=1)
