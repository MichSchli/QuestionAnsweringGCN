import numpy as np
import tensorflow as tf

class CellUpdate:

    def __init__(self, prefix, in_dimension, out_dimension, graph):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = prefix
        self.graph = graph

        self.entity_cell = IndividualCell(prefix+"entity", in_dimension, out_dimension)
        self.event_cell = IndividualCell(prefix+"event", in_dimension, out_dimension)

    def update(self, entity_context, event_context):
        entity_update, entity_cell_update = self.entity_cell.get_update(self.graph.entity_vertex_embeddings,
                                                                        entity_context,
                                                                        self.graph.entity_cell_state)


        event_update, event_cell_update = self.event_cell.get_update(self.graph.event_vertex_embeddings,
                                                                     event_context,
                                                                     self.graph.event_cell_state)

        self.graph.event_cell_state = event_cell_update
        self.graph.entity_cell_state = entity_cell_update

        self.graph.entity_vertex_embeddings = entity_update
        self.graph.event_vertex_embeddings = event_update

    def prepare_tensorflow_variables(self, mode="train"):
        self.entity_cell.prepare_tensorflow_variables(mode=mode)
        self.event_cell.prepare_tensorflow_variables(mode=mode)

    def get_regularization_term(self):
        return self.entity_cell.get_regularization_term() + self.event_cell.get_regularization_term()

class IndividualCell:

    def __init__(self, prefix, in_dimension, out_dimension):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = prefix

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

    def prepare_tensorflow_variables(self, mode="train"):
        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension + self.out_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_cell_update = tf.Variable(initializer_v, name=self.variable_prefix + "W_cell_update")
        self.b_cell_update = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "b_cell_update")

        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension + self.out_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_forget_gate = tf.Variable(initializer_v, name=self.variable_prefix + "W_forget_gate")
        self.b_forget_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "b_forget_gate")

        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension + self.out_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_write_gate = tf.Variable(initializer_v, name=self.variable_prefix + "W_write_gate")
        self.b_write_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "b_write_gate")

        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension + self.out_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_read_gate = tf.Variable(initializer_v, name=self.variable_prefix + "W_read_gate")
        self.b_read_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "b_read_gate")

    def get_regularization_term(self):
        return 0.0