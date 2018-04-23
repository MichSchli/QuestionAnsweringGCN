import tensorflow as tf


class Gcn:

    def __init__(self, messages, gates, updater, graph):
        self.variables = {}
        self.messages = messages
        self.gates = gates
        self.updater = updater
        self.graph = graph

    def propagate(self, mode, previous_carry_over):
        message_sums = self.compute_message_sums(mode)
        carry_over = self.updater.update(message_sums, previous_carry_over)

        return carry_over

    def compute_message_sums(self, mode):
        sender_indices, receiver_indices = self.graph.get_edges()
        incidence_matrix = self.get_unnormalized_incidence_matrix(receiver_indices)

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
        return self.messages.get_regularization() + self.gates.get_regularization() + self.updater.get_regularization()