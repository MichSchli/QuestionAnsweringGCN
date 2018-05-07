import tensorflow as tf
import numpy as np


class CellStateGcnInitializer:

    def __init__(self, prefix, in_dimension, out_dimension, graph):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = prefix
        self.graph = graph

        self.prepare_tensorflow_variables()

    def initialize(self, mode):
        initial_evidence = self.graph.get_full_vertex_embeddings()

        write_gate = tf.nn.sigmoid(tf.matmul(initial_evidence, self.W_write_gate) + self.b_write_gate)
        cell_update = tf.nn.tanh(tf.matmul(initial_evidence, self.W_cell_update) + self.b_cell_update)
        read_gate = tf.nn.sigmoid(tf.matmul(initial_evidence, self.W_read_gate) + self.b_read_gate)

        cell_state = write_gate * cell_update

        output = read_gate * tf.nn.tanh(cell_state)

        self.graph.update_vertex_embeddings(output)

        return cell_state

    def prepare_tensorflow_variables(self):
        initializer_v = np.random.normal(0, 0.01,
                                         size=(self.in_dimension, self.out_dimension)).astype(
            np.float32)
        self.W_cell_update = tf.Variable(initializer_v, name=self.variable_prefix + "W_cell_update")
        self.b_cell_update = tf.Variable(np.zeros(self.out_dimension).astype(np.float32),
                                         name=self.variable_prefix + "b_cell_update")

        initializer_v = np.random.normal(0, 0.01,
                                         size=(self.in_dimension, self.out_dimension)).astype(
            np.float32)
        self.W_write_gate = tf.Variable(initializer_v, name=self.variable_prefix + "W_write_gate")
        self.b_write_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32),
                                         name=self.variable_prefix + "b_write_gate")

        initializer_v = np.random.normal(0, 0.01,
                                         size=(self.in_dimension, self.out_dimension)).astype(
            np.float32)
        self.W_read_gate = tf.Variable(initializer_v, name=self.variable_prefix + "W_read_gate")
        self.b_read_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32),
                                         name=self.variable_prefix + "b_read_gate")



class CellStateGcnUpdater:
    def __init__(self, prefix, in_dimension, out_dimension, graph):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = prefix
        self.graph = graph

        self.prepare_tensorflow_variables()

    def update(self, new_observations, previous_cell_state):
        previous_vectors = self.graph.get_embeddings()

        update, new_cell_state = self.get_update(previous_vectors, new_observations, previous_cell_state)
        self.graph.update_vertex_embeddings(update)

        return new_cell_state

    def get_update(self, previous_vectors, context_vectors, previous_cell_state):
        combined_input = tf.concat([previous_vectors, context_vectors], -1)

        write_gate = tf.nn.sigmoid(tf.matmul(combined_input, self.W_write_gate) + self.b_write_gate)
        read_gate = tf.nn.sigmoid(tf.matmul(combined_input, self.W_read_gate) + self.b_read_gate)

        cell_update = tf.nn.tanh(tf.matmul(combined_input, self.W_cell_update) + self.b_cell_update)
        new_cell_state = write_gate * cell_update

        if previous_cell_state is not None:
            forget_gate = tf.nn.sigmoid(tf.matmul(combined_input, self.W_forget_gate) + self.b_forget_gate)
            new_cell_state += forget_gate * previous_cell_state

        output = read_gate * tf.nn.tanh(new_cell_state)

        return output, new_cell_state

    def prepare_tensorflow_variables(self):
        initializer_v = np.random.normal(0, 0.01,
                                         size=(self.in_dimension + self.out_dimension, self.out_dimension)).astype(
            np.float32)
        self.W_cell_update = tf.Variable(initializer_v, name=self.variable_prefix + "W_cell_update")
        self.b_cell_update = tf.Variable(np.zeros(self.out_dimension).astype(np.float32),
                                         name=self.variable_prefix + "b_cell_update")

        initializer_v = np.random.normal(0, 0.01,
                                         size=(self.in_dimension + self.out_dimension, self.out_dimension)).astype(
            np.float32)
        self.W_forget_gate = tf.Variable(initializer_v, name=self.variable_prefix + "W_forget_gate")
        self.b_forget_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32),
                                         name=self.variable_prefix + "b_forget_gate")

        initializer_v = np.random.normal(0, 0.01,
                                         size=(self.in_dimension + self.out_dimension, self.out_dimension)).astype(
            np.float32)
        self.W_write_gate = tf.Variable(initializer_v, name=self.variable_prefix + "W_write_gate")
        self.b_write_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32),
                                        name=self.variable_prefix + "b_write_gate")

        initializer_v = np.random.normal(0, 0.01,
                                         size=(self.in_dimension + self.out_dimension, self.out_dimension)).astype(
            np.float32)
        self.W_read_gate = tf.Variable(initializer_v, name=self.variable_prefix + "W_read_gate")
        self.b_read_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32),
                                       name=self.variable_prefix + "b_read_gate")

    def get_regularization(self):
        return 0