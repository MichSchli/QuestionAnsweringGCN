import numpy as np
import tensorflow as tf

from candidate_selection.tensorflow_models.abstract_tensorflow_model import AbstractTensorflowModel
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


class EntityEmbeddingVsLstm(AbstractTensorflowModel):

    def get_preprocessor_stack_types(self):
        preprocessor_stack_types = ["hypergraph", "gold", "sentence"]
        if self.model_settings["static_entity_embeddings"]:
            preprocessor_stack_types += ["static_entity_embeddings"]
        return preprocessor_stack_types


    def set_indexers(self, indexers):
        self.word_indexer = indexers.word_indexer
        self.entity_indexer = indexers.entity_indexer

    def initialize_graph(self):
        if not self.model_settings["static_entity_embeddings"]:
            self.entity_embedding = VectorEmbedding(self.entity_indexer, self.variables, variable_prefix="entity")
            self.add_component(self.entity_embedding)
        else:
            self.entity_embedding = StaticVectorEmbedding(self.entity_indexer, self.variables, variable_prefix="entity")
            self.add_component(self.entity_embedding)

        self.word_embedding = SequenceEmbedding(self.word_indexer, self.variables, variable_prefix="word")
        self.add_component(self.word_embedding)

        self.lstms = [BiLstm(self.variables, self.model_settings["word_embedding_dimension"], variable_prefix="lstm_" + str(i)) for i in
                      range(self.model_settings["n_lstms"])]
        for lstm in self.lstms:
            self.add_component(lstm)

        self.lstm_attention = BiLstm(self.variables, self.model_settings["word_embedding_dimension"], variable_prefix="lstm_attention")
        self.add_component(self.lstm_attention)

        self.target_comparator = TargetComparator(self.variables, variable_prefix="comparison_to_sentence")
        self.add_component(self.target_comparator)

        self.decoder = SoftmaxDecoder(self.variables)
        self.add_component(self.decoder)

        if self.model_settings["use_transformation"]:
            self.transformation = MultilayerPerceptron([self.model_settings["word_embedding_dimension"],
                                                        self.model_settings["entity_embedding_dimension"]],
                                                       self.variables,
                                                       variable_prefix="transformation",
                                                       l2_scale=self.model_settings["regularization_scale"])
            self.add_component(self.transformation)


    def OLD_initialize_indexers(self):
        self.word_indexer = self.build_indexer(self.model_settings["word_embedding_type"], (40000, self.model_settings["word_dimension"]), self.model_settings["default_word_embedding"])
        self.entity_indexer = self.build_indexer(self.model_settings["entity_embedding_type"],
                                                 (self.model_settings["facts"].number_of_entities,
                                                  self.model_settings["entity_dimension"]),
                                                 self.model_settings["default_entity_embedding"])
        self.relation_indexer = self.build_indexer(self.model_settings["relation_embedding_type"],
                                                   (self.model_settings["facts"].number_of_relation_types,
                                                    self.model_settings["entity_dimension"]),
                                                   self.model_settings["default_relation_embedding"])


    def compute_entity_scores(self):
        entity_scores = self.entity_embedding.get_representations()
        word_scores = self.word_embedding.get_representations()

        for lstm in self.lstms:
            word_scores = lstm.transform_sequences(word_scores)

        attention_scores = self.lstm_attention.transform_sequences(word_scores)
        attention_values = tf.nn.softmax(attention_scores, dim=1)
        attention_weighted_word_scores = word_scores * attention_values
        target_vector = tf.reduce_sum(attention_weighted_word_scores, 1)

        if self.model_settings["use_transformation"]:
            target_vector = self.transformation.transform(target_vector)

        entity_scores = self.target_comparator.get_comparison_scores(target_vector,
                                                                     entity_scores)

        return entity_scores