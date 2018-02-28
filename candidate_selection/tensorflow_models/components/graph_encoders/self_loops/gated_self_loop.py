import numpy as np
import tensorflow as tf

class GatedSelfLoop:

    def __init__(self, prefix, in_dimension, out_dimension):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = prefix

    def get_update(self, vectors):
        update = tf.nn.relu(tf.matmul(vectors, self.W) + self.b)
        gate = tf.nn.sigmoid(tf.matmul(vectors, self.W_gate) + self.b_gate)

        return update * gate

    def prepare_tensorflow_variables(self):
        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(
                np.float32)
        self.W = tf.Variable(initializer_v, name=self.variable_prefix + "self_gate_weights")

        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_gate = tf.Variable(initializer_v, name=self.variable_prefix + "self_weights")

        self.b = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "self_bias")
        self.b_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "self_gate_bias")