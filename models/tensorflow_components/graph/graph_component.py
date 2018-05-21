import tensorflow as tf
import numpy as np

from example_reader.graph_reader.edge_type_utils import EdgeTypeUtils
from models.tensorflow_components.graph.assignment_view import AssignmentView


class GraphComponent:

    vertex_embeddings = None
    variables = None
    variable_assignments = None

    mention_dummy_assignment_view = None

    def __init__(self):
        self.variables = {}
        self.define_variable(tf.int32, "n_vertices")
        self.define_variable(tf.int32, "entity_vertex_indexes")
        self.define_variable(tf.int32, "sender_indices")
        self.define_variable(tf.int32, "receiver_indices")
        self.define_variable(tf.int32, "edge_type_indices")
        self.define_variable(tf.int32, "edge_bow_matrix")
        self.define_variable(tf.float32, "mention_scores")
        self.define_variable(tf.float32, "vertex_types")
        self.define_variable(tf.int32, "sentence_vertex_indices")
        self.define_variable(tf.int32, "word_vertex_indices")
        self.define_variable(tf.float32, "vertex_max_scores")

        self.edge_type_utils = EdgeTypeUtils()
        for i in range(self.edge_type_utils.count_types()):
            self.variables["gcn_types_"+str(i)] = tf.placeholder(tf.int32, name="gcn_types_"+str(i))

        self.mention_dummy_assignment_view = AssignmentView()
        self.word_assignment_view = AssignmentView()

    def define_variable(self, type, name):
        self.variables[name] = tf.placeholder(type, name=name)

    def initialize_zero_embeddings(self, dimension):
        self.vertex_embeddings = tf.zeros((self.get_variable("n_vertices"), dimension))

    def initialize_dummy_counts(self):
        mention_scores = self.get_variable("mention_scores")
        self.vertex_embeddings = self.mention_dummy_assignment_view.get_all_vectors(tf.expand_dims(mention_scores,-1))

    def set_dummy_embeddings(self, embeddings):
        self.vertex_embeddings += self.mention_dummy_assignment_view.get_all_vectors(embeddings)

    def set_word_embeddings(self, embeddings, reshape=True):
        if reshape:
            shape = embeddings.get_shape().as_list()
            embeddings = tf.reshape(embeddings, [-1, shape[-1]])
        self.vertex_embeddings += self.word_assignment_view.get_all_vectors(embeddings)

    def get_sentence_embeddings(self):
        return tf.nn.embedding_lookup(self.vertex_embeddings, self.get_variable("sentence_vertex_indices"))

    def get_word_vertex_embeddings(self):
        return tf.nn.embedding_lookup(self.vertex_embeddings, self.get_variable("word_vertex_indices"))

    def get_dummy_counts(self):
        return self.get_variable("mention_scores")

    def get_vertex_max_scores(self):
        return self.get_variable("vertex_max_scores")

    def get_full_vertex_embeddings(self):
        features = [self.get_embeddings(),
                    self.get_vertex_types(),
                    tf.expand_dims(self.get_vertex_max_scores(), -1)]

        features = tf.concat(features, -1)
        return features

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

    def get_edges(self, edge_type):
        if edge_type is None:
            return self.variables["sender_indices"], self.variables["receiver_indices"]
        else:
            return tf.nn.embedding_lookup(self.variables["sender_indices"], self.variables["gcn_types_"+str(edge_type)]), \
                   tf.nn.embedding_lookup(self.variables["receiver_indices"], self.variables["gcn_types_"+str(edge_type)])

    def get_gcn_type_edge_indices(self, edge_type):
        return self.variables["gcn_types_"+str(edge_type)]

    def get_edge_types(self, edge_type):
        if edge_type is None:
            return self.variables["edge_type_indices"]
        else:
            return tf.nn.embedding_lookup(self.variables["edge_type_indices"], self.variables["gcn_types_"+str(edge_type)])

    def get_edge_word_bows(self, edge_type):
        if edge_type is None:
            return self.variables["edge_bow_matrix"]
        else:
            return tf.nn.embedding_lookup(self.variables["edge_bow_matrix"], self.variables["gcn_types_"+str(edge_type)])

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
        self.variable_assignments["vertex_max_scores"] = batch.get_max_score_by_vertex()

        for i in range(self.edge_type_utils.count_types()):
            self.variable_assignments["gcn_types_"+str(i)] = batch.get_combined_gcn_type_edge_indices(i)

        mention_dummy_indices = batch.get_combined_mention_dummy_indices()
        total_vertices = batch.count_all_vertices()
        self.mention_dummy_assignment_view.assign(mention_dummy_indices, total_vertices, np.arange(mention_dummy_indices.shape[0], dtype=np.int32))

        word_vertex_indices = batch.get_combined_word_vertex_indices()
        word_row_vertices = batch.get_word_indexes_in_flattened_sentence_matrix()
        self.word_assignment_view.assign(word_vertex_indices, total_vertices, word_row_vertices)

    def get_regularization(self):
        return 0