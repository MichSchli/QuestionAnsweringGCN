import tensorflow as tf
import numpy as np

from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


class GcnConcatMessagePasser(AbstractComponent):

    facts = None
    variables = None
    random = None
    dimension = None
    variable_prefix = None

    n_coefficients = 1
    submatrix_d = None

    senders = None
    receivers = None
    use_inverse_edges_instead = None

    def __init__(self, facts, variables, dimension, variable_prefix="", senders="events", receivers="entities", inverse_edges=False, weights="block", biases="constant", gate_mode="none"):
        self.facts = facts
        self.variables = variables
        self.dimension = dimension
        self.submatrix_d = int(dimension / self.n_coefficients)
        self.weight_type = weights
        self.bias_type = biases

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

        self.senders = senders
        self.receivers = receivers
        self.use_inverse_edges_instead = inverse_edges
        if gate_mode != "none":
            self.use_gates = True
        self.gate_mode = gate_mode

    def set_gate_features(self, features):
        self.gate_features = features

    def get_update(self, hypergraph):
        sender_indices, receiver_indices = hypergraph.get_edges(senders=self.senders, receivers=self.receivers, inverse_edges=self.use_inverse_edges_instead)
        types = hypergraph.get_edge_types(senders=self.senders, receivers=self.receivers, inverse_edges=self.use_inverse_edges_instead)

        sender_embeddings = hypergraph.get_embeddings(self.senders)
        receiver_embeddings = hypergraph.get_embeddings(self.receivers)

        event_to_entity_matrix = self.get_globally_normalized_incidence_matrix(receiver_indices, tf.shape(receiver_embeddings)[0])
        #event_to_entity_matrix = self.get_locally_normalized_incidence_matrix(receiver_indices,
        #                                                                      types,
        #                                                                      tf.shape(receiver_embeddings)[0])

        messages = tf.nn.embedding_lookup(sender_embeddings, sender_indices)
        if self.use_gates and self.gate_mode == "features_given":
            gate_features = tf.nn.embedding_lookup(self.gate_features, sender_indices)
            gate_values = tf.matmul(gate_features, self.gate_transform) + self.gate_bias
            gates = tf.nn.sigmoid(gate_values)

        ###
        if self.weight_type == "blocks":
            transformations = tf.nn.embedding_lookup(self.W, types)
            reshape_embeddings = tf.reshape(messages, [-1, self.n_coefficients, self.submatrix_d])
            transformed_messages = tf.squeeze(tf.matmul(transformations, tf.expand_dims(reshape_embeddings, -1)))
            transformed_messages = tf.reshape(transformed_messages, [-1, self.dimension])
            messages = transformed_messages

        if self.bias_type == "constant":
            messages += self.b
        elif self.bias_type == "relation_specific":
            type_biases = tf.nn.embedding_lookup(self.b, types)
            messages += type_biases

        if self.use_gates:
            messages = messages * gates

        sent_messages = tf.sparse_tensor_dense_matmul(event_to_entity_matrix, messages)
        return sent_messages

    def get_unnormalized_incidence_matrix(self, receiver_indices, number_of_receivers):
        mtr_values = tf.to_float(tf.ones_like(receiver_indices))

        message_count = tf.shape(receiver_indices)[0]
        message_indices = tf.range(message_count, dtype=tf.int32)

        mtr_indices = tf.to_int64(tf.transpose(tf.stack([receiver_indices, message_indices])))
        mtr_shape = tf.to_int64(tf.stack([number_of_receivers, message_count]))

        tensor = tf.SparseTensor(indices=mtr_indices,
                                 values=mtr_values,
                                 dense_shape=mtr_shape)

        return tensor

    def get_globally_normalized_incidence_matrix(self, receiver_indices, number_of_receivers):
        mtr_values = tf.to_float(tf.ones_like(receiver_indices))

        message_count = tf.shape(receiver_indices)[0]
        message_indices = tf.range(message_count, dtype=tf.int32)

        mtr_indices = tf.to_int64(tf.transpose(tf.stack([receiver_indices, message_indices])))
        mtr_shape = tf.to_int64(tf.stack([number_of_receivers, message_count]))

        tensor = tf.sparse_softmax(tf.SparseTensor(indices=mtr_indices,
                                 values=mtr_values,
                                 dense_shape=mtr_shape))

        return tensor

    def get_locally_normalized_incidence_matrix(self, receiver_indices, message_types, number_of_receivers):
        mtr_values = tf.to_float(tf.ones_like(receiver_indices))

        message_count = tf.shape(receiver_indices)[0]
        message_indices = tf.range(message_count, dtype=tf.int32)

        mtr_indices = tf.to_int64(tf.transpose(tf.stack([message_types, receiver_indices, message_indices])))

        mtr_shape = tf.to_int64(tf.stack([self.facts.number_of_relation_types, number_of_receivers, message_count]))

        tensor = tf.sparse_softmax(tf.SparseTensor(indices=mtr_indices,
                                                   values=mtr_values,
                                                   dense_shape=mtr_shape))

        return tf.sparse_reduce_sum_sparse(tensor, 0)

    def prepare_variables(self):
        if self.use_gates:
            n_gate_features = 1
            initializer = np.random.normal(0, 0.01, size=(n_gate_features, self.dimension)).astype(np.float32)
            self.gate_transform = tf.Variable(initializer, name=self.variable_prefix + "weights")
            self.gate_bias = tf.Variable(np.zeros(self.dimension).astype(np.float32))

        if self.weight_type == "blocks":
            initializer = np.random.normal(0, 0.01, size=(self.facts.number_of_relation_types, self.n_coefficients, self.submatrix_d, self.submatrix_d)).astype(np.float32)
            self.W = tf.Variable(initializer, name=self.variable_prefix + "weights")

        if self.bias_type == "constant":
            self.b = tf.Variable(np.zeros(self.dimension).astype(np.float32))
        elif self.bias_type == "relation_specific":
            self.b = tf.Variable(np.random.normal(0, 1, (self.facts.number_of_relation_types, self.dimension)).astype(np.float32))
