import tensorflow as tf

from candidate_selection.tensorflow_models.abstract_tensorflow_model import AbstractTensorflowModel
from candidate_selection.tensorflow_models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.tensorflow_models.components.embeddings.sequence_embedding import SequenceEmbedding
from candidate_selection.tensorflow_models.components.embeddings.static_vector_embedding import StaticVectorEmbedding
from candidate_selection.tensorflow_models.components.embeddings.vector_embedding import VectorEmbedding
from candidate_selection.tensorflow_models.components.extras.target_comparator import TargetComparator
from candidate_selection.tensorflow_models.components.vector_encoders.multilayer_perceptron import MultilayerPerceptron
from input_models.hypergraph.hypergraph_preprocessor import HypergraphPreprocessor
from input_models.mask.mask_preprocessor import LookupMaskPreprocessor
from input_models.sentences.sentence_preprocessor import SentencePreprocessor
from input_models.static_embedding.static_entity_embedding_preprocessor import StaticEntityEmbeddingPreprocessor


class EntityEmbeddingVsBagOfWords(AbstractTensorflowModel):

    hypergraph_batch_preprocessor = None

    def initialize_graph(self):
        if not self.model_settings["static_entity_embeddings"]:
            self.entity_embedding = VectorEmbedding(self.entity_indexer, self.variables, variable_prefix="entity")
            self.add_component(self.entity_embedding)
        else:
            self.entity_embedding = StaticVectorEmbedding(self.entity_indexer, self.variables, variable_prefix="entity")
            self.add_component(self.entity_embedding)

        self.word_embedding = SequenceEmbedding(self.word_indexer, self.variables, variable_prefix="word")
        self.add_component(self.word_embedding)

        self.target_comparator = TargetComparator(self.variables, variable_prefix="comparison_to_sentence")
        self.add_component(self.target_comparator)

        self.decoder = SoftmaxDecoder(self.variables)
        self.add_component(self.decoder)

        if self.model_settings["use_transformation"]:
            self.transformation = MultilayerPerceptron([self.model_settings["word_dimension"],
                                                        self.model_settings["entity_dimension"]],
                                                       self.variables,
                                                       variable_prefix="transformation")
            self.add_component(self.transformation)

    def initialize_preprocessors(self):
        self.hypergraph_batch_preprocessor = HypergraphPreprocessor(self.entity_indexer, self.relation_indexer,
                                                                    "neighborhood", "neighborhood_input_model", None)
        self.preprocessor = LookupMaskPreprocessor("neighborhood_input_model", "entity_vertex_matrix", "gold_entities",
                                                   "gold_mask", self.hypergraph_batch_preprocessor)
        self.preprocessor = SentencePreprocessor(self.word_indexer, "sentence", "question_sentence_input_model",
                                                 self.preprocessor)

        if self.model_settings["static_entity_embeddings"]:
            self.preprocessor = StaticEntityEmbeddingPreprocessor(self.entity_indexer, "neighborhood_input_model", self.preprocessor)


    def initialize_indexers(self):
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
        bag_of_words = tf.reduce_sum(word_scores, 1)

        if self.model_settings["use_transformation"]:
            bag_of_words = self.transformation.transform(bag_of_words)

        entity_scores = self.target_comparator.get_comparison_scores(bag_of_words,
                                                                     entity_scores)

        return entity_scores

    def retrieve_entities(self, graph_index, entity_index):
        return [self.hypergraph_batch_preprocessor.retrieve_entity_labels_in_batch(graph_index, entity_index)]
