import tensorflow as tf


class AssignmentView:

    def __init__(self):
        self.variables = {}
        self.variables["total_vertex_count"] = tf.placeholder(tf.int32)
        self.variables["vertex_indices"] = tf.placeholder(tf.int32)
        self.variables["from_range"] = tf.placeholder(tf.int32)

    def get_all_vectors(self, view_embeddings):
        from_size = tf.shape(view_embeddings)[0]
        from_indices = self.variables["from_range"]

        to_size = self.variables["total_vertex_count"]
        to_indices = self.variables["vertex_indices"]
        values = tf.ones_like(from_indices, dtype=tf.float32)

        stacked_indices = tf.transpose(tf.stack([to_indices, from_indices]))
        indices = tf.to_int64(stacked_indices)
        shape = tf.to_int64([to_size, from_size])

        matrix = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
        return tf.sparse_tensor_dense_matmul(matrix, view_embeddings)

    def assign(self, target_indices, total_vertex_count, from_range):
        self.variable_assignments = {}
        self.variable_assignments["total_vertex_count"] = total_vertex_count
        self.variable_assignments["vertex_indices"] = target_indices
        self.variable_assignments["from_range"] = from_range

    def get_regularization(self):
        return 0

    def handle_variable_assignment(self, batch, mode):
        pass