import tensorflow as tf


class BiLstm:

    variables = None
    dimension = None
    variable_prefix = None

    stored = None

    def __init__(self, in_dimension, out_dimension, variable_prefix):
        self.variables = {}
        self.variables["sentence_lengths"] = tf.placeholder(tf.int32, [None], name=variable_prefix+"lengths")

        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = variable_prefix

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}
        self.variable_assignments["sentence_lengths"] = batch.get_sentence_lengths()

    def transform_sequences(self, sequences):
        if self.stored is None:
            cell_forward = tf.contrib.rnn.LSTMCell(self.in_dimension, num_proj=self.out_dimension/2)
            cell_backward = tf.contrib.rnn.LSTMCell(self.in_dimension, num_proj=self.out_dimension/2)
            lengths = self.variables["sentence_lengths"]
            self.stored = tf.concat(tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, sequences, dtype=tf.float32, sequence_length=lengths)[0], -1)

        return self.stored