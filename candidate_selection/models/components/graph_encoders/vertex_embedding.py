import tensorflow as tf
import numpy as np

class VertexEmbedding:

    facts = None
    variables = None
    random = None
    dimension = None

    def __init__(self, facts, variables, dimension, random=False, variable_prefix=""):
        self.facts = facts
        self.variables = variables
        self.random = random
        self.dimension = dimension

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def get_representations(self):
        if self.random:
            return self.get_random_representation()
        else:
            return self.get_embedding()

    def get_random_representation(self):
        print("Warning: Random embedding used.")
        return tf.random_normal((self.variables.get_variable(self.variable_prefix+"number_of_elements_in_batch"), self.dimension))

    def get_embedding(self):
        return tf.nn.embedding_lookup(self.W, self.variables.get_variable(self.variable_prefix+"element_indices"))

    def prepare_variables(self):
        if self.random:
            self.variables.add_variable(self.variable_prefix+"number_of_elements_in_batch", tf.placeholder(tf.int32, name=self.variable_prefix+"number_of_elements_in_batch"))
        else:
            self.variables.add_variable(self.variable_prefix+"element_indices", tf.placeholder(tf.int32, name=self.variable_prefix+"element_indices"))
            initializer = np.random.normal(0, 1, size=(self.facts.number_of_entities, self.dimension)).astype(np.float32)
            self.W = tf.Variable(initializer)

    def handle_variable_assignment(self, variable):
        if self.random:
            self.variables.assign_variable(self.variable_prefix+"number_of_elements_in_batch", variable)
        else:
            self.variables.assign_variable(self.variable_prefix+"element_indices", variable)