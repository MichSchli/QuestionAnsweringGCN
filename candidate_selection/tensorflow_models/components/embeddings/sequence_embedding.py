import tensorflow as tf


class SequenceEmbedding:

    variables = None
    random = None
    dimension = None

    def __init__(self, indexer, variables, variable_prefix=""):
        self.indexer = indexer
        self.variables = variables

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

        initializer = self.indexer.get_all_vectors()

        self.W = tf.Variable(initializer)

    def handle_variable_assignment(self, sentence_input_model):
        self.variables.assign_variable(self.variable_prefix+"indices", sentence_input_model.word_index_matrix)
        self.variables.assign_variable(self.variable_prefix+"mask", sentence_input_model.word_index_matrix > 0)
