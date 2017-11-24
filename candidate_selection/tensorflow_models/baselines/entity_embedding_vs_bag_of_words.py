import tensorflow as tf

from candidate_selection.tensorflow_hypergraph_representation import TensorflowHypergraphRepresentation
from candidate_selection.tensorflow_models.abstract_tensorflow_model import AbstractTensorflowModel
from candidate_selection.tensorflow_models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.tensorflow_models.components.embeddings.sequence_embedding import SequenceEmbedding
from candidate_selection.tensorflow_models.components.embeddings.static_vector_embedding import StaticVectorEmbedding
from candidate_selection.tensorflow_models.components.embeddings.vector_embedding import VectorEmbedding
from candidate_selection.tensorflow_models.components.extras.embedding_retriever import EmbeddingRetriever
from candidate_selection.tensorflow_models.components.extras.target_comparator import TargetComparator
from candidate_selection.tensorflow_models.components.vector_encoders.multilayer_perceptron import MultilayerPerceptron
from candidate_selection.tensorflow_sentence_representation import TensorflowSentenceRepresentation


class EntityEmbeddingVsBagOfWords(AbstractTensorflowModel):


    def get_preprocessor_stack_types(self):
        preprocessor_stack_types = ["hypergraph", "gold", "sentence"]
        if self.model_settings["static_entity_embeddings"]:
            preprocessor_stack_types += ["static_entity_embeddings"]
        return preprocessor_stack_types

    def initialize_graph(self):
        if not self.model_settings["static_entity_embeddings"]:
            self.entity_embedding = VectorEmbedding(self.entity_indexer, self.variables, variable_prefix="entity")
            self.add_component(self.entity_embedding)
        else:
            self.entity_embedding = StaticVectorEmbedding(self.entity_indexer, self.variables, variable_prefix="entity")
            self.add_component(self.entity_embedding)

        self.hypergraph = TensorflowHypergraphRepresentation(self.variables)
        self.add_component(self.hypergraph)

        self.question_sentence = TensorflowSentenceRepresentation(self.variables)
        self.add_component(self.question_sentence)

        self.word_embedding = SequenceEmbedding(self.word_indexer, self.variables, variable_prefix="word")
        self.add_component(self.word_embedding)

        self.target_comparator = TargetComparator(self.variables, variable_prefix="comparison_to_sentence")
        self.add_component(self.target_comparator)

        self.decoder = SoftmaxDecoder(self.variables)
        self.add_component(self.decoder)

        self.centroid_to_sentence_mapper = EmbeddingRetriever(self.variables, duplicate_policy="sum")
        self.add_component(self.centroid_to_sentence_mapper)

        if self.model_settings["use_transformation"]:
            self.transformation = MultilayerPerceptron([self.model_settings["word_embedding_dimension"],
                                                        self.model_settings["entity_embedding_dimension"]],
                                                       self.variables,
                                                       variable_prefix="transformation",
                                                       l2_scale=self.model_settings["regularization_scale"])


            self.centroid_transformation = MultilayerPerceptron([self.model_settings["word_embedding_dimension"],
                                                                 self.model_settings["entity_embedding_dimension"]],
                                                                self.variables,
                                                                variable_prefix="centroid_transformation",
                                                                l2_scale=self.model_settings["regularization_scale"])
            self.add_component(self.centroid_transformation)
            self.add_component(self.transformation)

    def set_indexers(self, indexers):
        self.word_indexer = indexers.word_indexer
        self.entity_indexer = indexers.entity_indexer

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
        self.hypergraph.entity_vertex_embeddings = self.entity_embedding.get_representations()
        word_embeddings = self.word_embedding.get_representations()

        centroid_embeddings = self.hypergraph.get_centroid_embeddings()
        centroid_embeddings = self.centroid_transformation.transform(centroid_embeddings)
        word_embeddings += self.centroid_to_sentence_mapper.add_to_embedding(self.hypergraph.entity_vertex_embeddings, centroid_embeddings, self.question_sentence.centroid_map)

        bag_of_words = tf.reduce_sum(word_embeddings, 1)

        if self.model_settings["use_transformation"]:
            bag_of_words = self.transformation.transform(bag_of_words)

        entity_embeddings = self.target_comparator.get_comparison_scores(bag_of_words,
                                                                     entity_embeddings)

        return entity_embeddings