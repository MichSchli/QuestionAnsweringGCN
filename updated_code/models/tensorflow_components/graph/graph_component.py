import tensorflow as tf
import numpy as np
from models.tensorflow_components.graph.assignment_view import AssignmentView


class GraphComponent:

    vertex_embeddings = None
    variables = None
    variable_assignments = None

    mention_dummy_assignment_view = None

    def __init__(self):
        self.variables = {}
        self.variables["n_vertices"] = tf.placeholder(tf.int32)
        self.variables["entity_vertex_indexes"] = tf.placeholder(tf.int32)
        self.variables["sender_indices"] = tf.placeholder(tf.int32)
        self.variables["receiver_indices"] = tf.placeholder(tf.int32)
        self.variables["edge_type_indices"] = tf.placeholder(tf.int32)
        self.variables["edge_bow_matrix"] = tf.placeholder(tf.int32)
        self.variables["mention_scores"] = tf.placeholder(tf.float32)
        self.variables["vertex_types"] = tf.placeholder(tf.float32)
        self.variables["sentence_vertex_indices"] = tf.placeholder(tf.int32)
        self.variables["word_vertex_indices"] = tf.placeholder(tf.int32)

        self.mention_dummy_assignment_view = AssignmentView()
        self.word_assignment_view = AssignmentView()

    def initialize_zero_embeddings(self, dimension):
        self.vertex_embeddings = tf.zeros((self.get_variable("n_vertices"), dimension))

    def initialize_dummy_counts(self):
        mention_scores = self.get_variable("mention_scores")
        self.vertex_embeddings = self.mention_dummy_assignment_view.get_all_vectors(tf.expand_dims(mention_scores,-1))

    def set_dummy_embeddings(self, embeddings):
        self.vertex_embeddings += self.mention_dummy_assignment_view.get_all_vectors(embeddings)

    def set_word_embeddings(self, embeddings):
        shape = embeddings.get_shape().as_list()
        self.vertex_embeddings += self.word_assignment_view.get_all_vectors(tf.reshape(embeddings, [-1, shape[-1]]))

    def get_sentence_embeddings(self):
        return tf.nn.embedding_lookup(self.vertex_embeddings, self.get_variable("sentence_vertex_indices"))

    def get_word_vertex_embeddings(self):
        return tf.nn.embedding_lookup(self.vertex_embeddings, self.get_variable("word_vertex_indices"))

    def get_dummy_counts(self):
        return self.get_variable("mention_scores")

    def get_embeddings(self):
        return self.vertex_embeddings

    def get_embedding_dimension(self):
        return self.embedding_dimension

    def update_vertex_embeddings(self, updated_embeddings):
        self.vertex_embeddings = updated_embeddings

    def get_target_vertex_embeddings(self):
        return tf.nn.embedding_lookup(self.vertex_embeddings, self.get_variable("entity_vertex_indexes"))

    def get_variable(self, name):
        return self.variables[name]

    def get_edges(self):
        return self.variables["sender_indices"], self.variables["receiver_indices"]

    def get_edge_types(self):
        return self.variables["edge_type_indices"]

    def get_edge_word_bows(self):
        return self.variables["edge_bow_matrix"]

    def count_vertices(self):
        return self.variables["n_vertices"]

    def get_vertex_types(self):
        return self.variables["vertex_types"]

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}
        self.variable_assignments["n_vertices"] = batch.count_all_vertices()
        self.variable_assignments["entity_vertex_indexes"] = batch.get_combined_entity_vertex_map_indexes()
        self.variable_assignments["sender_indices"] = batch.get_combined_sender_indices()
        self.variable_assignments["receiver_indices"] = batch.get_combined_receiver_indices()
        self.variable_assignments["edge_type_indices"] = batch.get_combined_edge_type_indices()
        self.variable_assignments["edge_bow_matrix"] = batch.get_padded_edge_part_type_matrix()
        self.variable_assignments["mention_scores"] = batch.get_combined_mention_scores()
        self.variable_assignments["vertex_types"] = batch.get_combined_vertex_types()
        self.variable_assignments["sentence_vertex_indices"] = batch.get_combined_sentence_vertex_indices()
        self.variable_assignments["word_vertex_indices"] = batch.get_combined_word_vertex_indices()

        mention_dummy_indices = batch.get_combined_mention_dummy_indices()
        total_vertices = batch.count_all_vertices()
        self.mention_dummy_assignment_view.assign(mention_dummy_indices, total_vertices, np.arange(mention_dummy_indices.shape[0], dtype=np.int32))

        word_vertex_indices = batch.get_combined_word_vertex_indices()
        word_row_vertices = batch.get_word_indexes_in_flattened_sentence_matrix()
        self.word_assignment_view.assign(word_vertex_indices, total_vertices, word_row_vertices)

    def get_regularization(self):
        return 0