import tensorflow as tf
import numpy as np
import math
from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


class BiLstm(AbstractComponent):

    variables = None
    dimension = None
    variable_prefix = None

    stored = None

    def __init__(self, variables, in_dimension, out_dimension, variable_prefix=""):
        self.variables = variables
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def handle_variable_assignment(self, batch_dictionary, mode):
        self.variables.assign_variable(self.variable_prefix+"lengths", batch_dictionary["question_sentence_input_model"].sentence_lengths)

    def prepare_tensorflow_variables(self, mode="train"):
        self.variables.add_variable(self.variable_prefix+"lengths", tf.placeholder(tf.int32, [None], name=self.variable_prefix+"lengths"))

    def transform_sequences(self, sequences):
        if self.stored is None:
            with tf.variable_scope(self.variable_prefix):
                cell_forward = tf.contrib.rnn.LSTMCell(self.in_dimension, num_proj=self.out_dimension/2)
                cell_backward = tf.contrib.rnn.LSTMCell(self.in_dimension, num_proj=self.out_dimension/2)
                lengths = self.variables.get_variable(self.variable_prefix+"lengths")
                self.stored = tf.concat(tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, sequences, dtype=tf.float32, sequence_length=lengths)[0], -1)

        return self.stored