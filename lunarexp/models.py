import tensorflow as tf


class Probnet(tf.keras.Model):
    def __init__(self, action_len, dense=50, supportsize=51, name="probnet"):
        super(Probnet, self).__init__(name=name)
        self.flat = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(dense, activation="relu")
        self.d2 = tf.keras.layers.Dense(dense, activation="relu")
        self.d3 = tf.keras.layers.Dense(dense, activation="relu")
        self.probs = ProbOutput(input_shape=(dense),
                                outdim=action_len,
                                gridsize=supportsize)

    @tf.function
    def call(self, x, training=False):
        y = self.flat(x)
        y = self.d1(y)
        y = self.d2(y)
        y = self.d3(y)
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
