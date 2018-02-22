import tensorflow as tf


class ReluTransform:

    def apply(self, vector):
        return tf.nn.relu(vector)

    def prepare_variables(self):
        pass