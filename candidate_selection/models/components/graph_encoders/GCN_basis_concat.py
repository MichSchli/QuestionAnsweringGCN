import tensorflow as tf


class GCNBasisConcat:

    facts = None
    variables = None
    random = None
    dimension = None

    def __init__(self, facts, variables, dimension):
        self.facts = facts
        self.variables = variables
        self.dimension = dimension

    def apply(self, entity_embeddings, event_embeddings):
        event_to_entity_edges = self.variables.get_variable("event_to_entity_edges")
        event_to_entity_types = self.variables.get_variable("event_to_entity_message_types")
        event_indices = event_to_entity_edges[:,0]
        entity_indices = event_to_entity_edges[:,1]
        event_to_entity_matrix = self.get_locally_normalized_incidence_matrix(entity_indices,
                                                                              event_to_entity_types,
                                                                              tf.shape(entity_embeddings)[0])

        #For now just add event embeddings to entity embeddings
        messages = tf.nn.embedding_lookup(event_embeddings, event_indices)
        sent_messages = tf.sparse_tensor_dense_matmul(event_to_entity_matrix, messages)
        entity_embeddings += sent_messages

        return entity_embeddings

    def get_locally_normalized_incidence_matrix(self, receiver_indices, message_types, number_of_receivers):
        mtr_values = tf.to_float(tf.ones_like(receiver_indices))

        message_count = tf.shape(receiver_indices)[0]
        message_indices = tf.range(message_count, dtype=tf.int32)

        mtr_indices = tf.to_int64(tf.transpose(tf.stack([message_types, receiver_indices, message_indices]))) #message_indices

        mtr_shape = tf.to_int64(tf.stack([self.facts.number_of_relation_types, number_of_receivers, message_count]))

        tensor = tf.sparse_softmax(tf.SparseTensor(indices=mtr_indices,
                                                   values=mtr_values,
                                                   dense_shape=mtr_shape))

        return tf.sparse_reduce_sum_sparse(tensor, 0)

    def prepare_variables(self):
        self.variables.add_variable("event_to_entity_edges", tf.placeholder(tf.int32))
        self.variables.add_variable("entity_to_event_edges", tf.placeholder(tf.int32))
        self.variables.add_variable("entity_to_entity_edges", tf.placeholder(tf.int32))
        self.variables.add_variable("event_to_entity_message_types", tf.placeholder(tf.int32))
        self.variables.add_variable("entity_to_event_message_types", tf.placeholder(tf.int32))
        self.variables.add_variable("entity_to_entity_message_types", tf.placeholder(tf.int32))

    def handle_variable_assignment(self, event_to_entity_edges, event_to_entity_message_types, entity_to_event_edges, entity_to_event_message_types, entity_to_entity_edges
                                   , entity_to_entity_message_types):

        self.variables.assign_variable("event_to_entity_edges", event_to_entity_edges)
        self.variables.assign_variable("entity_to_event_edges", entity_to_event_edges)
        self.variables.assign_variable("entity_to_entity_edges", entity_to_entity_edges)
        self.variables.assign_variable("event_to_entity_message_types", event_to_entity_message_types)
        self.variables.assign_variable("entity_to_event_message_types", entity_to_event_message_types)
        self.variables.assign_variable("entity_to_entity_message_types", entity_to_entity_message_types)
