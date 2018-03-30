import tensorflow as tf
import numpy as np


class SentenceBatchFeatures:

    vectors_by_example = None

    def __init__(self, graph, width):
        self.graph = graph
        self.width = width
        self.variables = {}
        self.variables["edge_to_example_map"] = tf.placeholder(tf.int32)

    def set_batch_features(self, vectors_by_example):
        self.vectors_by_example = vectors_by_example

    def get(self):
        return tf.nn.embedding_lookup(self.vectors_by_example, self.variables["edge_to_example_map"])

    def get_width(self):
        return self.width

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}
        self.variable_assignments["edge_to_example_map"] = np.concatenate([np.repeat(i, example.count_edges()) for i,example in enumerate(batch.examples)])

    def prepare_variables(self):
        pass

    def get_regularization(self):
        return 0