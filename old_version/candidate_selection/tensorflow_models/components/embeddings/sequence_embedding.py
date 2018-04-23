import tensorflow as tf

from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


class SequenceEmbedding(AbstractComponent):

    variables = None
    random = None
    dimension = None
    word_dropout_rate = None

    def __init__(self, indexer, variables, variable_prefix="", word_dropout_rate=0.2, is_static=False):
        self.indexer = indexer
        self.variables = variables
        self.word_dropout_rate=word_dropout_rate
        self.is_static = is_static

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def get_representations(self, mode="train"):
        return self.get_embedding(mode=mode)

    def get_embedding(self, mode="train"):
        embedding = tf.nn.embedding_lookup(self.W, self.variables.get_variable(self.variable_prefix+"indices"))
        if mode == "train" and self.word_dropout_rate > 0.0:
            embedding_shape = tf.shape(embedding)
            noise_shape = [embedding_shape[0], embedding_shape[1], 1]
            embedding = tf.nn.dropout(embedding, 1-self.word_dropout_rate, noise_shape=noise_shape)
        return embedding * tf.expand_dims(self.variables.get_variable(self.variable_prefix+"mask"), -1)

    def prepare_tensorflow_variables(self, mode="train"):
        self.variables.add_variable(self.variable_prefix+"indices", tf.placeholder(tf.int32, [None, None], name=self.variable_prefix+"indices"))
        self.variables.add_variable(self.variable_prefix+"mask", tf.placeholder(tf.float32, [None, None], name=self.variable_prefix+"mask"))

        initializer = self.indexer.get_all_vectors()

        self.W = tf.Variable(initializer, trainable=not self.is_static)

    def handle_variable_assignment(self, batch_dictionary, mode):
        self.variables.assign_variable(self.variable_prefix+"indices", batch_dictionary["question_sentence_input_model"].word_index_matrix)
        self.variables.assign_variable(self.variable_prefix+"mask", batch_dictionary["question_sentence_input_model"].word_index_matrix > 0)
