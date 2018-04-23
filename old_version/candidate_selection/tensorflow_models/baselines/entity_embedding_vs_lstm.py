import tensorflow as tf

from candidate_selection.tensorflow_models.abstract_tensorflow_model import AbstractTensorflowModel
from candidate_selection.tensorflow_models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.tensorflow_models.components.embeddings.sequence_embedding import SequenceEmbedding
from candidate_selection.tensorflow_models.components.embeddings.static_vector_embedding import StaticVectorEmbedding
from candidate_selection.tensorflow_models.components.embeddings.vector_embedding import VectorEmbedding
from candidate_selection.tensorflow_models.components.extras.embedding_retriever import EmbeddingRetriever
from candidate_selection.tensorflow_models.components.extras.target_comparator import TargetComparator
from candidate_selection.tensorflow_models.components.sequence_encoders.bilstm import BiLstm
from candidate_selection.tensorflow_models.components.vector_encoders.multilayer_perceptron import MultilayerPerceptron


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

        self.target_comparator = TargetComparator(self.variables, variable_prefix="comparison_to_sentence")
        self.add_component(self.target_comparator)

        self.decoder = SoftmaxDecoder(self.variables)
        self.add_component(self.decoder)

        self.sentence_to_graph_mapper = EmbeddingRetriever(self.variables, duplicate_policy="sum", variable_prefix="mapper")
        self.add_component(self.sentence_to_graph_mapper)

        if self.model_settings["use_transformation"]:
            self.transformation = MultilayerPerceptron([self.model_settings["word_embedding_dimension"],
                                                        self.model_settings["entity_embedding_dimension"]],
                                                       self.variables,
                                                       variable_prefix="transformation",
                                                       l2_scale=self.model_settings["regularization_scale"])

            self.centroid_transformation = MultilayerPerceptron([self.model_settings["entity_embedding_dimension"],
                                                                 self.model_settings["word_embedding_dimension"]],
                                                                self.variables,
                                                                variable_prefix="centroid_transformation",
                                                                l2_scale=self.model_settings["regularization_scale"])
            self.add_component(self.centroid_transformation)
            self.add_component(self.transformation)


    def compute_entity_scores(self):
        entity_vertex_embeddings = self.entity_embedding.get_representations()
        word_embeddings = self.word_embedding.get_representations()
        word_embedding_shape = tf.shape(word_embeddings)
        word_embeddings = tf.reshape(word_embeddings, [-1, self.model_settings["word_embedding_dimension"]])

        centroid_embeddings = self.sentence_to_graph_mapper.get_forward_embeddings(entity_vertex_embeddings)
        centroid_embeddings = self.centroid_transformation.transform(centroid_embeddings)
        word_embeddings += self.sentence_to_graph_mapper.map_backwards(centroid_embeddings)
        word_embeddings = tf.reshape(word_embeddings, word_embedding_shape)

        for lstm in self.lstms:
            word_embeddings = lstm.transform_sequences(word_embeddings)

        attention_scores = self.lstm_attention.transform_sequences(word_embeddings)
        attention_values = tf.nn.softmax(attention_scores, dim=1)
        attention_weighted_word_scores = word_embeddings * attention_values
        target_vector = tf.reduce_sum(attention_weighted_word_scores, 1)

        if self.model_settings["use_transformation"]:
            target_vector = tf.nn.relu(target_vector)
            target_vector = self.transformation.transform(target_vector)

        entity_scores = self.target_comparator.get_comparison_scores(target_vector,
                                                                     entity_vertex_embeddings)

        return entity_scores