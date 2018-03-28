import tensorflow as tf
import numpy as np


class SentenceToEntityMapper:

    variables = None
    variable_assignment = None
    comparison_type = None

    def __init__(self, comparison_type="concat"):
        self.variables = {}
        self.variables["target_indices"] = tf.placeholder(tf.int32)
        self.comparison_type = comparison_type

    def map(self, sentence_embeddings, word_embeddings):
        distributed_sentence_embeddings = tf.nn.embedding_lookup(sentence_embeddings, self.variables["target_indices"])

        if self.comparison_type == "dot_product":
            comparison = tf.reduce_sum(distributed_sentence_embeddings * word_embeddings, axis=1)
        elif self.comparison_type == "sum":
            comparison = distributed_sentence_embeddings + word_embeddings
        elif self.comparison_type == "concat":
            comparison = tf.concat([distributed_sentence_embeddings, word_embeddings], axis=1)

        return comparison

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}
        self.variable_assignments["target_indices"] = np.concatenate([np.repeat(i, example.count_entities()) for i, example in enumerate(batch.examples)])
