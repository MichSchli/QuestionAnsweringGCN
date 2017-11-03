import numpy as np
import tensorflow as tf

from candidate_selection.tensorflow_models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.tensorflow_models.components.embeddings.static_vector_embedding import StaticVectorEmbedding
from candidate_selection.tensorflow_models.components.extras.target_comparator import TargetComparator
from candidate_selection.tensorflow_models.components.sequence_encoders.bilstm import BiLstm

from candidate_selection.tensorflow_models.components.embeddings.sequence_embedding import SequenceEmbedding
from candidate_selection.tensorflow_models.components.embeddings.vector_embedding import VectorEmbedding
from candidate_selection.tensorflow_models.components.vector_encoders.multilayer_perceptron import MultilayerPerceptron
from candidate_selection.tensorflow_variables_holder import TensorflowVariablesHolder
from helpers.static import Static
from indexing.freebase_indexer import FreebaseIndexer
from indexing.glove_indexer import GloveIndexer
from indexing.lazy_indexer import LazyIndexer
from input_models.hypergraph.hypergraph_preprocessor import HypergraphPreprocessor
from input_models.mask.mask_preprocessor import LookupMaskPreprocessor
from input_models.sentences.sentence_preprocessor import SentencePreprocessor
from input_models.static_embedding.static_entity_embedding_preprocessor import StaticEntityEmbeddingPreprocessor


class EntityEmbeddingVsLstm:

    decoder = None
    variables = None
    facts = None
    entity_embedding = None
    hypergraph_batch_preprocessor = None
    entity_scores = None
    sentence_iterator = None

    is_tensorflow = True

    entity_dimension = None
    embedding_dimension = None
    use_transformation = None
    n_lstms = None

    word_embedding_type = None
    entity_embedding_type = None
    relation_embedding_type = None

    default_word_embedding = None
    default_entity_embedding = None
    default_relation_embedding = None

    def update_setting(self, setting_string, value):
        if setting_string == "dimension":
            self.entity_dimension = int(value)
            self.word_dimension = int(value)
        elif setting_string == "word_dimension":
            self.word_dimension = int(value)
        elif setting_string == "entity_dimension":
            self.entity_dimension = int(value)
        if setting_string == "n_lstms":
            self.n_lstms = int(value)
        elif setting_string == "word_embedding_type":
            self.word_embedding_type = value
        elif setting_string == "entity_embedding_type":
            self.entity_embedding_type = value
        elif setting_string == "relation_embedding_type":
            self.relation_embedding_type = value
        elif setting_string == "default_word_embedding":
            self.default_word_embedding = value
        elif setting_string == "default_entity_embedding":
            self.default_entity_embedding = value
        elif setting_string == "default_relation_embedding":
            self.default_relation_embedding = value
        elif setting_string == "facts":
            self.facts = value
        elif setting_string == "static_entity_embeddings":
            self.static_entity_embeddings = True if value == "True" else False
        elif setting_string == "use_transformation":
            self.use_transformation = True if value == "True" else False

    def build_indexer(self, string, shape, default_embedding):
        if string is None or string == "none":
            return LazyIndexer(shape)
        elif string == "initialized" and default_embedding == "GloVe":
            key = default_embedding + "_" + str(shape[1])
            if key not in Static.embedding_indexers:
                print("building")
                Static.embedding_indexers[key] = GloveIndexer(shape[1])
            return Static.embedding_indexers[key]
        elif string == "initialized" and default_embedding == "Siva":
            key = default_embedding
            if key not in Static.embedding_indexers:
                Static.embedding_indexers[key] = FreebaseIndexer()
            return Static.embedding_indexers[key]

    def initialize(self):
        self.initialize_indexers()
        self.initialize_preprocessors()
        self.initialize_graph()

    def initialize_graph(self):
        self.variables = TensorflowVariablesHolder()

        if not self.static_entity_embeddings:
            self.entity_embedding = VectorEmbedding(self.entity_indexer, self.variables, variable_prefix="entity")
        else:
            self.entity_embedding = StaticVectorEmbedding(self.entity_indexer, self.variables, variable_prefix="entity")

        self.word_embedding = SequenceEmbedding(self.word_indexer, self.variables, variable_prefix="word")

        self.lstms = [BiLstm(self.variables, self.word_dimension, variable_prefix="lstm_" + str(i)) for i in
                      range(self.n_lstms)]
        self.lstm_attention = BiLstm(self.variables, self.word_dimension, variable_prefix="lstm_attention")

        self.target_comparator = TargetComparator(self.variables, variable_prefix="comparison_to_sentence")
        self.decoder = SoftmaxDecoder(self.variables)

        if self.use_transformation:
            self.transformation = MultilayerPerceptron([self.word_dimension, self.entity_dimension], self.variables, variable_prefix="transformation")


    def initialize_preprocessors(self):
        self.hypergraph_batch_preprocessor = HypergraphPreprocessor(self.entity_indexer, self.relation_indexer,
                                                                    "neighborhood", "neighborhood_input_model", None)
        self.preprocessor = LookupMaskPreprocessor("neighborhood_input_model", "entity_vertex_matrix", "gold_entities",
                                                   "gold_mask", self.hypergraph_batch_preprocessor)
        self.preprocessor = SentencePreprocessor(self.word_indexer, "sentence", "question_sentence_input_model",
                                                 self.preprocessor)

        if self.static_entity_embeddings:
            self.preprocessor = StaticEntityEmbeddingPreprocessor(self.entity_indexer, "neighborhood_input_model", self.preprocessor)

    def initialize_indexers(self):
        self.word_indexer = self.build_indexer(self.word_embedding_type, (40000, self.word_dimension), self.default_word_embedding)
        self.entity_indexer = self.build_indexer(self.entity_embedding_type,
                                                 (self.facts.number_of_entities, self.entity_dimension), self.default_entity_embedding)
        self.relation_indexer = self.build_indexer(self.relation_embedding_type,
                                                   (self.facts.number_of_relation_types, self.entity_dimension), self.default_relation_embedding)


    def get_preprocessor(self):
        return self.preprocessor

    def validate_example(self, example):
        candidates = example["neighborhood"].get_vertices(type="entities")
        target_vertices = example["gold_entities"]

        # For now, eliminate all elements without 100 % overlap. Otherwise, train loss is undefined.
        # Remember: This is also executed at test time, so will need refactoring
        target_vertices_in_candidates = np.isin(target_vertices, candidates)

        return target_vertices_in_candidates.all()

    def prepare_tensorflow_variables(self, mode="train"):
        self.entity_embedding.prepare_tensorflow_variables(mode=mode)
        self.word_embedding.prepare_tensorflow_variables(mode=mode)
        self.decoder.prepare_tensorflow_variables(mode=mode)
        self.target_comparator.prepare_tensorflow_variables()

        self.lstm_attention.prepare_tensorflow_variables(mode=mode)

        for lstm in self.lstms:
            lstm.prepare_tensorflow_variables(mode=mode)

        if self.use_transformation:
            self.transformation.prepare_tensorflow_variables()


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

        for lstm in self.lstms:
            word_scores = lstm.transform_sequences(word_scores)

        attention_scores = self.lstm_attention.transform_sequences(word_scores)
        attention_values = tf.nn.softmax(attention_scores, dim=1)
        attention_weighted_word_scores = word_scores * attention_values
        target_vector = tf.reduce_sum(attention_weighted_word_scores, 1)

        if self.use_transformation:
            target_vector = self.transformation.transform(target_vector)

        entity_scores = self.target_comparator.get_comparison_scores(target_vector,
                                                                     entity_scores)

        return entity_scores

    def handle_variable_assignment(self, o_preprocessed_batch, mode="predict"):
        hypergraph_input_model = o_preprocessed_batch["neighborhood_input_model"]
        if not self.static_entity_embeddings:
            self.entity_embedding.handle_variable_assignment(hypergraph_input_model.entity_map)
        else:
            self.entity_embedding.handle_variable_assignment(hypergraph_input_model)

        self.word_embedding.handle_variable_assignment(o_preprocessed_batch["question_sentence_input_model"])
        self.target_comparator.handle_variable_assignment(hypergraph_input_model.get_instance_indices())

        self.decoder.handle_variable_assignment(hypergraph_input_model.entity_vertex_matrix, hypergraph_input_model.entity_vertex_slices)
        self.decoder.assign_gold_variable(o_preprocessed_batch["gold_mask"])

        self.lstm_attention.handle_variable_assignment(o_preprocessed_batch["question_sentence_input_model"])
        for lstm in self.lstms:
            lstm.handle_variable_assignment(o_preprocessed_batch["question_sentence_input_model"])

        return self.variables.get_assignment_dict()

    def retrieve_entities(self, graph_index, entity_index):
        return [self.hypergraph_batch_preprocessor.retrieve_entity_labels_in_batch(graph_index, entity_index)]
