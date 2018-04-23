import tensorflow as tf
import numpy as np

from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


class VectorEmbedding(AbstractComponent):

    variables = None
    random = None
    dimension = None

    W = None

    def __init__(self, indexer, variables, variable_prefix=""):
        self.indexer = indexer
        self.dimension = self.indexer.get_dimension()
        self.variables = variables

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def get_representations(self):
        return tf.nn.embedding_lookup(self.W, self.variables.get_variable(self.variable_prefix+"element_indices"))

    def prepare_tensorflow_variables(self, mode='train'):
        self.variables.add_variable(self.variable_prefix+"element_indices", tf.placeholder(tf.int32, [None], name=self.variable_prefix+"element_indices"))
        initializer = self.indexer.get_all_vectors()
        self.W = tf.Variable(initializer)

    def handle_variable_assignment(self, variable, mode):
        self.variables.assign_variable(self.variable_prefix+"element_indices", variable["neighborhood_input_model"].entity_map)