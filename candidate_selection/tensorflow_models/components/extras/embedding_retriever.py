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

    def from_embedding(self, embeddings, indices):
        return tf.nn.embedding_lookup(embeddings, indices)

    def to_embedding(self, values, indices):
        if self.duplicate_policy == "average":
            pass
        elif self.duplicate_policy == "sum":
            from_indices = self.variables.get_variable(
                tf.range(self.variables.get_variable(self.variable_prefix + "n_centroids")))
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
        self.variables.add_variable(self.variable_prefix + "target_embedding_size", tf.placeholder(tf.int32, shape=[None, None]))

    def handle_variable_assignment(self, batch, mode):
        self.variables.assign_variable(self.variable_prefix + "target_embedding_size", batch["neighborhood"].n_entities)