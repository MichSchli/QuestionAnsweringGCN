import tensorflow as tf


class GcnPropagator:

    def __init__(self, messages, gates, graph, direction, edge_type=None):
        self.variables = {}
        self.messages = messages
        self.gates = gates
        self.graph = graph
        self.direction = direction
        self.edge_type = edge_type

    def propagate(self, mode):
        message_sums = self.compute_message_sums(mode)

        return message_sums

    def compute_message_sums(self, mode):
        sender_indices, receiver_indices = self.graph.get_edges(edge_type=self.edge_type)

        message_in_indices = receiver_indices if self.direction == "forward" else sender_indices
        incidence_matrix = self.get_unnormalized_incidence_matrix(message_in_indices)

        messages = self.messages.get_messages(mode)
        gates = self.gates.get_gates(mode)
        messages = messages * gates

        sent_messages = tf.sparse_tensor_dense_matmul(incidence_matrix, messages)
        return sent_messages

    def get_unnormalized_incidence_matrix(self, receiver_indices):
        mtr_values = tf.to_float(tf.ones_like(receiver_indices))

        message_count = tf.shape(receiver_indices)[0]
        message_indices = tf.range(message_count, dtype=tf.int32)

        mtr_indices = tf.to_int64(tf.transpose(tf.stack([receiver_indices, message_indices])))
        mtr_shape = tf.to_int64(tf.stack([self.graph.count_vertices(), message_count]))

        tensor = tf.SparseTensor(indices=mtr_indices,
                                 values=mtr_values,
                                 dense_shape=mtr_shape)

        return tensor

    def handle_variable_assignment(self, batch, mode):
        pass

    def get_regularization(self):
        return self.messages.get_regularization() + self.gates.get_regularization()