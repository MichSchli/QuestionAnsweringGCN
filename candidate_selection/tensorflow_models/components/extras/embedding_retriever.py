from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent
import tensorflow as tf


class EmbeddingRetriever(AbstractComponent):

    variable_prefix = None
    variables = None
    duplicate_policy = None

    def __init__(self, variables, duplicate_policy="average", variable_prefix=""):
        self.variables = variables
        self.variable_prefix = variable_prefix
        self.duplicate_policy = duplicate_policy
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def get_forward_embeddings(self, embeddings):
        return tf.nn.embedding_lookup(embeddings, self.variables.get_variable(self.variable_prefix + "forward_indices"))

    def get_backward_embeddings(self, embeddings):
        return tf.nn.embedding_lookup(embeddings, self.variables.get_variable(self.variable_prefix + "backward_indices"))

    def map_backwards(self, temporary_embeddings):
        if self.duplicate_policy == "average":
            pass
        elif self.duplicate_policy == "sum":
            from_indices = tf.range(tf.shape(temporary_embeddings)[0])
            to_indices = self.variables.get_variable(self.variable_prefix + "backward_indices")
            values = tf.ones_like(from_indices, dtype=tf.float32)

            from_size = tf.shape(temporary_embeddings)[0]
            to_size = self.variables.get_variable(self.variable_prefix + "backward_total_size")

            indices = tf.to_int64(tf.transpose(tf.stack([to_indices, from_indices])))
            shape = tf.to_int64([to_size, from_size])

            matrix = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
            return tf.sparse_tensor_dense_matmul(matrix, temporary_embeddings)

    def OLD_to_embedding(self, values, indices):
        if self.duplicate_policy == "average":
            pass
        elif self.duplicate_policy == "sum":
            from_indices = tf.range(self.variables.get_variable(self.variable_prefix + "n_centroids"))
            to_indices = indices
            stg_values = tf.to_float(tf.ones_like(from_indices))

            from_size = self.variables.get_variable(self.variable_prefix + "n_centroids")
            to_size = self.variables.get_variable(self.variable_prefix + "target_embedding_size")

            stg_indices = tf.to_int64(tf.transpose(tf.stack([from_indices, to_indices])))
            stg_shape = tf.to_int64([from_size, to_size])

            matrix = tf.sparse_softmax(tf.SparseTensor(indices=stg_indices,
                                                     values=stg_values,
                                                     dense_shape=stg_shape))

            return tf.sparse_tensor_dense_matmul(matrix, values)
        else:
            pass

    def prepare_tensorflow_variables(self, mode="train"):
        self.variables.add_variable(self.variable_prefix + "forward_indices", tf.placeholder(tf.int32, shape=[None], name=self.variable_prefix + "forward_indices"))
        self.variables.add_variable(self.variable_prefix + "backward_indices", tf.placeholder(tf.int32, shape=[None], name=self.variable_prefix + "backward_indices"))
        self.variables.add_variable(self.variable_prefix + "forward_total_size", tf.placeholder(tf.int32, shape=None, name=self.variable_prefix + "forward_total_size"))
        self.variables.add_variable(self.variable_prefix + "backward_total_size", tf.placeholder(tf.int32, shape=None, name=self.variable_prefix + "backward_total_size"))

    def handle_variable_assignment(self, batch, mode):
        map = batch["sentence_to_neighborhood_map"]
        self.variables.assign_variable(self.variable_prefix + "forward_indices", map.flat_forward_map)
        self.variables.assign_variable(self.variable_prefix + "backward_indices", map.flat_backward_map)
        self.variables.assign_variable(self.variable_prefix + "forward_total_size", map.forward_total_size)
        self.variables.assign_variable(self.variable_prefix + "backward_total_size", map.backward_total_size)
