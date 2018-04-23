import numpy as np
import tensorflow as tf

class HighwayUpdate:

    def __init__(self, prefix, in_dimension, out_dimension, graph):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = prefix
        self.graph = graph

    def update(self, entity_context, event_context):
        entity_self_loop_messages = tf.nn.relu(tf.matmul(self.graph.entity_vertex_embeddings, self.W_en) + self.b_en)
        event_self_loop_messages = tf.nn.relu(tf.matmul(self.graph.event_vertex_embeddings, self.W_ev) + self.b_ev)

        entity_gate = tf.nn.sigmoid(tf.matmul(tf.concat([entity_context, self.graph.entity_vertex_embeddings], -1), self.W_en_gate) + self.b_en_gate)
        event_gate = tf.nn.sigmoid(tf.matmul(tf.concat([event_context, self.graph.event_vertex_embeddings], -1), self.W_ev_gate) + self.b_ev_gate)

        self.graph.entity_vertex_embeddings = entity_gate * entity_context + (1-entity_gate) * entity_self_loop_messages
        self.graph.event_vertex_embeddings = event_gate * event_context + (1-event_gate) * event_self_loop_messages

    def prepare_tensorflow_variables(self, mode='train'):
        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_en = tf.Variable(initializer_v, name=self.variable_prefix + "self_entity_weights")
        self.b_en = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "self_entity_bias")

        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_ev = tf.Variable(initializer_v, name=self.variable_prefix + "self_event_weights")
        self.b_ev = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "self_event_bias")

        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension+self.out_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_en_gate = tf.Variable(initializer_v, name=self.variable_prefix + "self_entity_weights_gate")
        self.b_en_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "self_entity_bias_gate")


        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension+self.out_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_ev_gate = tf.Variable(initializer_v, name=self.variable_prefix + "self_event_weights_gate")
        self.b_ev_gate = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "self_event_bias_gate")


    def get_regularization_term(self):
        return 0.0