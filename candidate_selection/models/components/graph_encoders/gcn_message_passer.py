import tensorflow as tf
import numpy as np


class GcnConcatMessagePasser:

    facts = None
    variables = None
    random = None
    dimension = None
    variable_prefix = None

    n_coefficients = 2
    submatrix_d = 3

    senders = None
    receivers = None
    inverse_edges = None

    def __init__(self, facts, variables, dimension, variable_prefix="", senders="events", receivers="entities", inverse_edges=False):
        self.facts = facts
        self.variables = variables
        self.dimension = dimension

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

        self.senders = senders
        self.receivers = receivers
        self.use_inverse_edges_instead = inverse_edges

    def get_update(self, hypergraph):
        sender_indices, receiver_indices = hypergraph.get_edges(senders=self.senders, receivers=self.receivers, inverse_edges=self.inverse_edges)
        types = hypergraph.get_edge_types(senders=self.senders, receivers=self.receivers, inverse_edges=self.inverse_edges)

        sender_embeddings = hypergraph.get_embeddings(self.senders)
        receiver_embeddings = hypergraph.get_embeddings(self.receivers)

        event_to_entity_matrix = self.get_locally_normalized_incidence_matrix(receiver_indices,
                                                                              types,
                                                                              tf.shape(receiver_embeddings)[0])

        #For now just add event embeddings to entity embeddings
        messages = tf.nn.embedding_lookup(sender_embeddings, sender_indices)

        ###
        transformations = tf.nn.embedding_lookup(self.W, types)
        reshape_embeddings = tf.reshape(messages, [-1, self.n_coefficients, self.submatrix_d])
        transformed_messages = tf.squeeze(tf.matmul(transformations, tf.expand_dims(reshape_embeddings, -1)))
        transformed_messages = tf.reshape(transformed_messages, [-1, 6])
        ###

        sent_messages = tf.sparse_tensor_dense_matmul(event_to_entity_matrix, transformed_messages)
        return sent_messages

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
        #TODO: W initializer is total crap. Also no bias.
        initializer = np.random.normal(0, 1, size=(self.facts.number_of_relation_types, self.n_coefficients, self.submatrix_d, self.submatrix_d)).astype(np.float32)
        self.W = tf.Variable(initializer, name=self.variable_prefix + "weights")
