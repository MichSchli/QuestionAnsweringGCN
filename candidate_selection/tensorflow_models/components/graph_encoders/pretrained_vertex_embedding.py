import tensorflow as tf
import numpy as np

class VertexEmbedding:

    facts = None
    variables = None
    random = None
    dimension = None

    def __init__(self, indexer, facts, variables, dimension, variable_prefix=""):
        self.indexer = indexer
        self.facts = facts
        self.variables = variables
        self.dimension = dimension

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def get_representations(self):
        return self.get_embedding()

    def get_random_representation(self):
        print("Warning: Random embedding used.")
        return tf.random_normal((self.variables.get_variable(self.variable_prefix+"number_of_elements_in_batch"), self.dimension))

    def get_embedding(self):
        return tf.nn.embedding_lookup(self.W, self.variables.get_variable(self.variable_prefix+"element_indices"))

    def prepare_tensorflow_variables(self, mode="train"):
        self.variables.add_variable(self.variable_prefix+"element_indices", tf.placeholder(tf.int32, [None], name=self.variable_prefix+"element_indices"))
        initializer = self.indexer.get_all_vectors()
        self.W = tf.Variable(initializer)

    def handle_variable_assignment(self, variable):
        self.variables.assign_variable(self.variable_prefix+"element_indices", variable)