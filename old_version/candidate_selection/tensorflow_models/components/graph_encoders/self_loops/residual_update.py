import numpy as np
import tensorflow as tf


class ResidualUpdate:

    def __init__(self, prefix, in_dimension, out_dimension, graph):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.graph = graph

        self.variable_prefix = prefix

    def update(self, entity_context, event_context):
        entity_self_loop_messages = tf.nn.relu(tf.matmul(self.graph.entity_vertex_embeddings, self.W_en) + self.b_en)
        event_self_loop_messages = tf.nn.relu(tf.matmul(self.graph.event_vertex_embeddings, self.W_ev) + self.b_ev)

        self.graph.entity_vertex_embeddings = entity_context + entity_self_loop_messages
        self.graph.event_vertex_embeddings = event_context + event_self_loop_messages

    def prepare_tensorflow_variables(self, mode='train'):
        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_en = tf.Variable(initializer_v, name=self.variable_prefix + "self_entity_weights")
        self.b_en = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "self_entity_bias")


        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_ev = tf.Variable(initializer_v, name=self.variable_prefix + "self_event_weights")
        self.b_ev = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "self_event_bias")

    def get_regularization_term(self):
        return 0.0