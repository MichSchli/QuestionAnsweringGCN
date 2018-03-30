import tensorflow as tf
import numpy as np
from models.tensorflow_components.transformations.multilayer_perceptron import MultilayerPerceptronComponent


class MultiheadAttention:

    heads = None
    query = None

    input_dimension = None
    variable_prefix = None
    attention_dropout = None

    def __init__(self, input_dimension, attention_heads=1, variable_prefix="", attention_dropout=0.0):
        self.input_dimension = input_dimension
        self.heads = attention_heads

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

        self.linear_key = MultilayerPerceptronComponent([int(0.5*self.input_dimension), int(0.5*self.input_dimension)], self.variable_prefix+"key_transform")
        self.linear_value = MultilayerPerceptronComponent([int(0.5*self.input_dimension), int(0.5*self.input_dimension)], self.variable_prefix+"value_transform")
        self.attention_dropout = attention_dropout

        self.variables = {}
        self.prepare_tensorflow_variables()

    def attend(self, padded_sequence_matrix, mode):
        key_matrix, value_matrix = tf.split(padded_sequence_matrix, [int(0.5*self.input_dimension),int(0.5*self.input_dimension)], 2)
        previous_shape = tf.shape(key_matrix)

        transformed_key = self.linear_key.transform(tf.reshape(key_matrix, [previous_shape[0]*previous_shape[1], -1]), mode)
        transformed_value = self.linear_value.transform(tf.reshape(value_matrix, [previous_shape[0]*previous_shape[1], -1]), mode)

        dim = int(0.5*self.input_dimension / self.heads)
        transformed_key = tf.reshape(transformed_key, [previous_shape[0], self.heads, previous_shape[1], dim])
        transformed_value = tf.reshape(transformed_value, [previous_shape[0], self.heads, previous_shape[1], dim])
        norm_factor = np.sqrt(dim)

        attention_weights = tf.nn.softmax(tf.reduce_sum(transformed_key * self.query, axis=3)/norm_factor, dim=-1)
        attention_weights = tf.expand_dims(attention_weights, 3)

        if mode=="train" and self.attention_dropout > 0.0:
            attention_weights = tf.nn.dropout(attention_weights, 1 - self.attention_dropout)

        attention_weighted_matrix = transformed_value*attention_weights

        weighted_value_matrix = tf.reduce_sum(attention_weighted_matrix, 2)
        return_value = tf.reshape(weighted_value_matrix, [previous_shape[0], -1])

        return return_value

    def prepare_tensorflow_variables(self):
        weight_initializer = np.random.uniform(-0.1, 0.1, size=(self.heads, 1, int(0.5*self.input_dimension/self.heads))).astype(np.float32)
        self.query = tf.Variable(weight_initializer, name=self.variable_prefix + "query")

    def handle_variable_assignment(self, batch, mode):
        pass

    def get_regularization(self):
        return 0