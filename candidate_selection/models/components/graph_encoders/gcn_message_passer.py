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

    def __init__(self, facts, variables, dimension, variable_prefix=""):
        self.facts = facts
        self.variables = variables
        self.dimension = dimension
        self.variable_prefix = ""

    def apply(self, entity_embeddings, event_embeddings):
        event_to_entity_edges = self.variables.get_variable("edges")
        event_to_entity_types = self.variables.get_variable("edge_types")
        event_indices = event_to_entity_edges[:,0]
        entity_indices = event_to_entity_edges[:,1]
        event_to_entity_matrix = self.get_locally_normalized_incidence_matrix(entity_indices,
                                                                              event_to_entity_types,
                                                                              tf.shape(entity_embeddings)[0])

        #For now just add event embeddings to entity embeddings
        messages = tf.nn.embedding_lookup(event_embeddings, event_indices)

        ###
        transformations = tf.nn.embedding_lookup(self.W, event_to_entity_types)
        reshape_embeddings = tf.reshape(messages, [-1, self.n_coefficients, self.submatrix_d])
        transformed_messages = tf.squeeze(tf.matmul(transformations, tf.expand_dims(reshape_embeddings, -1)))
        transformed_messages = tf.reshape(transformed_messages, [-1, 6])
        ###

        sent_messages = tf.sparse_tensor_dense_matmul(event_to_entity_matrix, transformed_messages)
        entity_embeddings += sent_messages

        return entity_embeddings

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
        self.variables.add_variable("edges", tf.placeholder(tf.int32))
        self.variables.add_variable("edge_types", tf.placeholder(tf.int32))

        #TODO: W is total crap
        initializer = np.random.normal(0, 1, size=(self.facts.number_of_relation_types, self.n_coefficients, self.submatrix_d, self.submatrix_d)).astype(np.float32)
        self.W = tf.Variable(initializer)

    def handle_variable_assignment(self, event_to_entity_edges, event_to_entity_message_types):
        self.variables.assign_variable("edges", event_to_entity_edges)
        self.variables.assign_variable("edge_types", event_to_entity_message_types)
