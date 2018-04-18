import tensorflow as tf


class WordPadder:

    def __init__(self):
        self.variables = {}

        self.prepare_tensorflow_variables()

    def prepare_tensorflow_variables(self):
        self.variables["word_indices"] = tf.placeholder(tf.int32, [None, None])

    def pad(self, word_embeddings):
        word_embeddings = tf.pad(word_embeddings, [[1,0], [0,0]], "CONSTANT")
        word_embedding_matrix = tf.nn.embedding_lookup(word_embeddings, self.variables["word_indices"])
        return word_embedding_matrix

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}

        self.variable_assignments["word_indices"] = batch.get_word_indexes_in_padded_sentence_matrix()

    def get_regularization(self):
        return 0