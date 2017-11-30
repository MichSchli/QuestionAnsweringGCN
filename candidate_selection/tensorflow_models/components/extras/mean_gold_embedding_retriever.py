from time import sleep

import tensorflow as tf
import numpy as np
from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


class MeanGoldEmbeddingRetriever(AbstractComponent):

    variable_prefix = None
    variables = None

    def __init__(self, variables, variable_prefix=""):
        self.variables = variables
        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def get_representations(self, embedding_matrix):
        entity_vertex_scores = tf.concat((tf.zeros([1, tf.shape(embedding_matrix)[1]], dtype=tf.float32), embedding_matrix), 0)
        entity_vertex_scores_distributed = tf.nn.embedding_lookup(entity_vertex_scores,
                                                                  self.variables.get_variable(self.variable_prefix + "gold_lookup_matrix"))

        entity_vertex_scores_distributed = tf.Print(entity_vertex_scores_distributed, [self.variables.get_variable(self.variable_prefix + "gold_lookup_matrix")], message="gold_lookup", summarize=25)

        #entity_vertex_scores_distributed = tf.Print(entity_vertex_scores_distributed, [entity_vertex_scores_distributed], message="distributed embs", summarize=30)
        return tf.reduce_sum(entity_vertex_scores_distributed, axis=1) \
               / tf.expand_dims(tf.to_float(tf.count_nonzero(self.variables.get_variable(self.variable_prefix + "gold_lookup_matrix"), axis=1)), axis=1)

    def prepare_tensorflow_variables(self, mode='predict'):
        self.variables.add_variable(self.variable_prefix + "gold_lookup_matrix", tf.placeholder(tf.int32))

    def handle_variable_assignment(self, batch_dictionary, mode):
        gold_matrix = batch_dictionary["gold_mask"]

        vertex_counts = batch_dictionary["neighborhood_input_model"].entity_vertex_slices
        indices = np.where(gold_matrix > 0)
        max_golds_in_example = np.max(np.bincount(indices[0]))
        matrix = np.zeros((gold_matrix.shape[0], max_golds_in_example), dtype=np.int32)
        pointer = np.zeros(gold_matrix.shape[0], dtype=np.int32)

        accumulator = 1
        prev = 0

        for row,value in zip(*indices):
            row_pointer = pointer[row]
            if row != prev:
                accumulator += vertex_counts[prev]
                prev = row
            pointer[row] += 1
            matrix[row, row_pointer] = value + accumulator

        self.variables.assign_variable(self.variable_prefix + "gold_lookup_matrix", matrix.astype(np.int32))
