from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent
import tensorflow as tf
import numpy as np


class Attention(AbstractComponent):

    query = None
    strategy = None
    input_dimension = None
    variable_prefix = None
    variables = None

    def __init__(self, input_dimension, variables, variable_prefix="", strategy=None):
        self.strategy = strategy
        self.input_dimension = input_dimension

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

        self.variables = variables

    def attend(self, padded_sequence_matrix):
        key_matrix, value_matrix = tf.split(padded_sequence_matrix, [int(0.5*self.input_dimension),int(0.5*self.input_dimension)], 2)
        norm_factor = np.sqrt(int(0.5*self.input_dimension)).astype(np.float32)
        attention_weights = tf.nn.softmax(tf.reduce_sum(key_matrix * self.query, axis=2)/norm_factor, dim=-1)

        return tf.reduce_sum(value_matrix*tf.expand_dims(attention_weights,2), 1)

    def prepare_tensorflow_variables(self, mode="train"):
        weight_initializer = np.random.uniform(-0.1, 0.1, size=(int(0.5*self.input_dimension))).astype(np.float32)
        self.query = tf.Variable(weight_initializer, name=self.variable_prefix + "_query")

