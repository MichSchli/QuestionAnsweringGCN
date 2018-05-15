import tensorflow as tf
import numpy as np


class SimpleSelfLoopGcnInitializer:

    def __init__(self, prefix, in_dimension, out_dimension, graph):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = prefix
        self.graph = graph

        self.prepare_tensorflow_variables()

    def initialize(self, mode):
        initial_evidence = self.graph.get_full_vertex_embeddings()
        initial_output = tf.nn.sigmoid(tf.matmul(initial_evidence, self.W) + self.b)
        self.graph.update_vertex_embeddings(initial_output)

        return None

    def prepare_tensorflow_variables(self):
        initializer_v = np.random.normal(0, 0.01,
                                         size=(self.in_dimension, self.out_dimension)).astype(
            np.float32)
        self.W = tf.Variable(initializer_v, name=self.variable_prefix + "W")
        self.b = tf.Variable(np.zeros(self.out_dimension).astype(np.float32),
                                         name=self.variable_prefix + "b")

class SimpleSelfLoopGcnUpdater:
    def __init__(self, prefix, in_dimension, out_dimension, graph):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = prefix
        self.graph = graph

        self.prepare_tensorflow_variables()

    def update(self, new_observations, previous_context):
        previous_vectors = self.graph.get_embeddings()
        combined_input = tf.concat([previous_vectors, new_observations], -1)
        output = tf.nn.sigmoid(tf.matmul(combined_input, self.W) + self.b)

        self.graph.update_vertex_embeddings(output)

        return None

    def prepare_tensorflow_variables(self):
        initializer_v = np.random.normal(0, 0.01,
                                         size=(self.in_dimension*2, self.out_dimension)).astype(
            np.float32)
        self.W = tf.Variable(initializer_v, name=self.variable_prefix + "W")
        self.b = tf.Variable(np.zeros(self.out_dimension).astype(np.float32),
                                         name=self.variable_prefix + "b")

    def get_regularization(self):
        return 0