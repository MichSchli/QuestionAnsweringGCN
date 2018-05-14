import tensorflow as tf


class GcnBiasOnlyMessages:

    biases = None

    def __init__(self, biases):
        self.biases = biases
        self.prepare_variables()

    def get_messages(self, mode):
        if len(self.biases) > 0:
            biases = tf.concat([f.get() for f in self.biases], -1)
        else:
            biases = 0

        return biases

    def prepare_variables(self):
        [f.prepare_variables() for f in self.biases]

    def get_regularization(self):
        return 0