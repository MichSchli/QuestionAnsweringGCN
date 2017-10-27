import tensorflow as tf
import numpy as np


class BiLstm:

    variables = None
    dimension = None
    variable_prefix = None

    W_forget = None
    W_input = None
    W_output = None
    W_update = None
    W_hidden = None

    def __init__(self, variables, dimension, variable_prefix=""):
        self.variables = variables
        self.dimension = dimension

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def get_optimizable_parameters(self):
        return [self.W_forget, self.W_input, self.W_output, self.W_update, self.W_hidden]

    def handle_variable_assignment(self, sentence_input_model):
        self.variables.assign_variable(self.variable_prefix+"lengths", sentence_input_model.sentence_lengths)

    def prepare_tensorflow_variables(self, mode="train"):
        self.variables.add_variable(self.variable_prefix+"lengths", tf.placeholder(tf.int32, [None], name=self.variable_prefix+"lengths"))

    def lstm_cell(self, previous, current):
        prev_h = previous[0]
        prev_c = previous[1]
        x = current

        input_to_lstm = tf.concat([x, prev_h], -1)

        forget_gate = tf.nn.sigmoid(tf.matmul(input_to_lstm, self.W_forget))
        input_gate = tf.nn.sigmoid(tf.matmul(input_to_lstm, self.W_input))
        output_gate = tf.nn.sigmoid(tf.matmul(input_to_lstm, self.W_output))

        cell_update = tf.nn.tanh(tf.matmul(input_to_lstm, self.W_update))
        new_cell_state = forget_gate * prev_c + input_gate * cell_update

        new_hidden_state = output_gate * tf.nn.tanh(tf.matmul(input_to_lstm, self.W_hidden))

        return new_hidden_state, new_cell_state

    def transform_sequences(self, sequences):
        with tf.variable_scope(self.variable_prefix):
            cell_forward = tf.contrib.rnn.LSTMCell(self.dimension, num_proj=self.dimension/2)
            cell_backward = tf.contrib.rnn.LSTMCell(self.dimension, num_proj=self.dimension/2)
            lengths = self.variables.get_variable(self.variable_prefix+"lengths")
            transformed_seqs = tf.concat(tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, sequences, dtype=tf.float32, sequence_length=lengths)[0], -1)

        #print([vs.name for vs in tf.all_variables()])

        #sequences = tf.transpose(sequences, perm=[1,0,2])
        #t_mask = tf.transpose(self.variables.get_variable(self.variable_prefix+"mask"))
        #transformed_seqs = tf.scan(self.lstm_cell, sequences, initializer=(initial_h, initial_c))[0]

        #inverse = tf.reverse(sequences, [0])
        #initial_h = tf.zeros_like(sequences[0])
        #initial_c = tf.zeros_like(sequences[0])
        #transformed_inverse = tf.scan(self.lstm_cell, inverse, initializer=(initial_h, initial_c))[0]
        #proper_inverse = tf.reverse(transformed_inverse, [0])

        #transformed_seqs = tf.transpose(transformed_seqs+proper_inverse, perm=[1,0,2])* tf.expand_dims(self.variables.get_variable(self.variable_prefix+"mask"), -1)
        return transformed_seqs