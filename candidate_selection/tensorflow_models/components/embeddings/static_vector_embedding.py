import tensorflow as tf
import numpy as np

from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


class StaticVectorEmbedding(AbstractComponent):

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
        return self.variables.get_variable(self.variable_prefix+"batch_embeddings")

    def prepare_tensorflow_variables(self, mode='train'):
        self.variables.add_variable(self.variable_prefix+"batch_embeddings", tf.placeholder(tf.float32, [None, self.dimension], name=self.variable_prefix+"batch_embeddings"))

    def handle_variable_assignment(self, batch_dictionary, mode):
        self.variables.assign_variable(self.variable_prefix+"batch_embeddings", batch_dictionary["neighborhood_input_model"].entity_embeddings)