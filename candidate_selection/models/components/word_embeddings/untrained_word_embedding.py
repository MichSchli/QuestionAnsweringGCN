import tensorflow as tf
import numpy as np

class UntrainedWordEmbedding:

    variables = None
    random = None
    dimension = None
    vocabulary_size = None

    def __init__(self, vocabulary_size, variables, dimension, random=False, variable_prefix=""):
        self.variables = variables
        self.random = random
        self.dimension = dimension
        self.vocabulary_size = vocabulary_size

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def get_representations(self):
        return self.get_embedding()

    def get_embedding(self):
        embedding = tf.nn.embedding_lookup(self.W, self.variables.get_variable(self.variable_prefix+"indices"))
        return embedding * tf.expand_dims(self.variables.get_variable(self.variable_prefix+"mask"), -1)

    def prepare_tensorflow_variables(self, mode="train"):
        self.variables.add_variable(self.variable_prefix+"indices", tf.placeholder(tf.int32, [None, None], name=self.variable_prefix+"indices"))
        self.variables.add_variable(self.variable_prefix+"mask", tf.placeholder(tf.float32, [None, None], name=self.variable_prefix+"mask"))
        initializer = np.random.normal(0, 1, size=(self.vocabulary_size, self.dimension)).astype(np.float32)
        self.W = tf.Variable(initializer)

    def handle_variable_assignment(self, sentence_input_model):
        self.variables.assign_variable(self.variable_prefix+"indices", sentence_input_model.word_index_matrix)
        self.variables.assign_variable(self.variable_prefix+"mask", sentence_input_model.word_index_matrix > 0)