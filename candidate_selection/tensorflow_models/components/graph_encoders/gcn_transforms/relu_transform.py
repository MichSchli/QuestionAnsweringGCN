import tensorflow as tf


class ReluTransform:

    def __init__(self, dimension):
        self.dimension = dimension

    def apply(self, vector):
        return tf.nn.relu(vector)

    def prepare_variables(self):
        pass