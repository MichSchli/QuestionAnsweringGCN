from candidate_selection.models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.models.components.extras.target_comparator import TargetComparator
from candidate_selection.models.components.graph_encoders.vertex_embedding import VertexEmbedding
from candidate_selection.tensorflow_variables_holder import TensorflowVariablesHolder
from input_models.hypergraph.hypergraph_preprocessor import HypergraphPreprocessor
import tensorflow as tf
import numpy as np


class DumbEntityEmbeddingModel:

    decoder = None
    variables = None
    facts = None
    entity_embedding = None
    hypergraph_batch_preprocessor = None
    entity_scores = None

    def __init__(self, facts, dimension):
        self.facts = facts
        self.dimension = dimension

        self.variables = TensorflowVariablesHolder()
        self.entity_embedding = VertexEmbedding(self.facts, self.variables, self.dimension,random=False)
        self.event_embedding = VertexEmbedding(self.facts, self.variables, self.dimension,random=True)
        self.decoder = SoftmaxDecoder(self.variables)

        self.hypergraph_batch_preprocessor = HypergraphPreprocessor()

    def get_aux_iterators(self):
        return []

    def validate_example(self, example):
        candidates = example[0].get_vertices(type="entities")
        target_vertices = example[-1]

        # For now, eliminate all elements without 100 % overlap. Otherwise, train loss is undefined.
        # Remember: This is also executed at test time, so will need refactoring
        target_vertices_in_candidates = np.isin(target_vertices, candidates)

        return target_vertices_in_candidates.all()

    def prepare_tensorflow_variables(self, mode="train"):
        self.entity_embedding.prepare_tensorflow_variables(mode=mode)
        self.event_embedding.prepare_tensorflow_variables(mode=mode)
        self.decoder.prepare_tensorflow_variables(mode=mode)

    def get_optimizable_parameters(self):
        optimizable_vars = self.entity_embedding.get_optimizable_parameters()
        return optimizable_vars

    def get_loss_graph(self, sum_examples=True):
        if self.entity_scores is None:
            self.entity_scores = self.compute_entity_scores()

        return self.decoder.decode_to_loss(self.entity_scores, sum_examples=sum_examples)

    def get_prediction_graph(self):
        if self.entity_scores is None:
            self.entity_scores = self.compute_entity_scores()

        return self.decoder.decode_to_prediction(self.entity_scores)

    def compute_entity_scores(self):
        scores = self.entity_embedding.get_representations()
        scores = tf.reduce_sum(scores, 1)
        return scores

    def handle_variable_assignment(self, o_preprocessed_batch, mode="predict"):
        hypergraph_input_model = o_preprocessed_batch[0]
        self.event_embedding.handle_variable_assignment(hypergraph_input_model.n_events)
        self.entity_embedding.handle_variable_assignment(hypergraph_input_model.entity_map)
        self.decoder.handle_variable_assignment(hypergraph_input_model.entity_vertex_matrix, hypergraph_input_model.entity_vertex_slices)

        self.decoder.assign_gold_variable(o_preprocessed_batch[-1])

        return self.variables.get_assignment_dict()

    def retrieve_entities(self, graph_index, entity_index):
        return [self.hypergraph_batch_preprocessor.retrieve_entity_labels_in_batch(graph_index, entity_index)]

    def preprocess(self, batch, mode='test'):
        hypergraph_input_model = self.hypergraph_batch_preprocessor.preprocess(batch[0])

        preprocessed = [hypergraph_input_model]

        gold_matrix = np.zeros_like(hypergraph_input_model.entity_vertex_matrix, dtype=np.float32)
        shitty_counter = 0
        for i, golds in enumerate(batch[-1]):
            gold_indexes = np.array([self.hypergraph_batch_preprocessor.retrieve_entity_indexes_in_batch(i,gold) for gold in golds])
            gold_matrix[i][gold_indexes - shitty_counter] = 1
            shitty_counter = np.max(hypergraph_input_model.entity_vertex_matrix[i])
        preprocessed += [gold_matrix]

        return preprocessed
