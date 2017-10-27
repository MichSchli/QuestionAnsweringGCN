from candidate_selection.models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.models.components.extras.target_comparator import TargetComparator
from candidate_selection.models.components.graph_encoders.vertex_embedding import VertexEmbedding
from candidate_selection.models.components.word_embeddings.untrained_word_embedding import UntrainedWordEmbedding
from candidate_selection.tensorflow_variables_holder import TensorflowVariablesHolder
from input_models.hypergraph.hypergraph_preprocessor import HypergraphPreprocessor
import tensorflow as tf
import numpy as np

from input_models.sentences.sentence_preprocessor import SentencePreprocessor


class LstmVsGcnModel:

    decoder = None
    variables = None
    facts = None
    entity_embedding = None
    hypergraph_batch_preprocessor = None
    entity_scores = None
    sentence_iterator = None

    def __init__(self, facts, dimension, sentence_iterator):
        self.facts = facts
        self.dimension = dimension
        self.sentence_iterator = sentence_iterator

        self.variables = TensorflowVariablesHolder()
        self.entity_embedding = VertexEmbedding(self.facts, self.variables, self.dimension,random=False, variable_prefix="entity")
        # Stupidly assume vocabulary size of 1000 (actually compute later)
        self.word_embedding = UntrainedWordEmbedding(1000, self.variables, self.dimension, random=False, variable_prefix="word")
        self.event_embedding = VertexEmbedding(self.facts, self.variables, self.dimension,random=True, variable_prefix="event")
        self.decoder = SoftmaxDecoder(self.variables)

        self.hypergraph_batch_preprocessor = HypergraphPreprocessor()
        self.sentence_batch_preprocessor = SentencePreprocessor()

        self.target_comparator = TargetComparator(self.variables, variable_prefix="comparison_to_sentence")

    def get_aux_iterators(self):
        return [self.sentence_iterator.get_iterator()]

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
        self.word_embedding.prepare_tensorflow_variables(mode=mode)
        self.decoder.prepare_tensorflow_variables(mode=mode)
        self.target_comparator.prepare_tensorflow_variables()

    def get_optimizable_parameters(self):
        optimizable_vars = self.entity_embedding.get_optimizable_parameters()
        optimizable_vars += self.word_embedding.get_optimizable_parameters()
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
        entity_scores = self.entity_embedding.get_representations()
        word_scores = self.word_embedding.get_representations()
        bag_of_words = tf.reduce_sum(word_scores, 1)

        entity_scores = self.target_comparator.get_comparison_scores(bag_of_words,
                                                                     entity_scores)

        return entity_scores

    def handle_variable_assignment(self, o_preprocessed_batch, mode="predict"):
        hypergraph_input_model = o_preprocessed_batch[0]
        self.event_embedding.handle_variable_assignment(hypergraph_input_model.n_events)
        self.entity_embedding.handle_variable_assignment(hypergraph_input_model.entity_map)

        self.word_embedding.handle_variable_assignment(o_preprocessed_batch[1])
        self.target_comparator.handle_variable_assignment(o_preprocessed_batch[2])

        self.decoder.handle_variable_assignment(hypergraph_input_model.entity_vertex_matrix, hypergraph_input_model.entity_vertex_slices)
        self.decoder.assign_gold_variable(o_preprocessed_batch[-1])

        return self.variables.get_assignment_dict()

    def retrieve_entities(self, graph_index, entity_index):
        return [self.hypergraph_batch_preprocessor.retrieve_entity_labels_in_batch(graph_index, entity_index)]


    def preprocess(self, batch, mode='test'):
        hypergraph_input_model = self.hypergraph_batch_preprocessor.preprocess(batch[0])
        sentence_input_model = self.sentence_batch_preprocessor.preprocess(batch[1])

        preprocessed = [hypergraph_input_model, sentence_input_model]

        targets = np.zeros(hypergraph_input_model.n_entities, dtype=np.int32)
        pointer = 0
        counter = 0
        for r in (hypergraph_input_model.entity_vertex_matrix != 0).sum(1):
            targets[pointer:pointer+r] = counter
            pointer += r
            counter += 1

        preprocessed += [targets]

        gold_matrix = np.zeros_like(hypergraph_input_model.entity_vertex_matrix, dtype=np.float32)
        shitty_counter = 0
        for i, golds in enumerate(batch[-1]):
            gold_indexes = np.array([self.hypergraph_batch_preprocessor.retrieve_entity_indexes_in_batch(i,gold) for gold in golds])
            gold_matrix[i][gold_indexes - shitty_counter] = 1
            shitty_counter = np.max(hypergraph_input_model.entity_vertex_matrix[i])
        preprocessed += [gold_matrix]

        return preprocessed
