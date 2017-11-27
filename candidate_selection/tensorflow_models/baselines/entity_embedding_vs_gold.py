import tensorflow as tf

from candidate_selection.tensorflow_hypergraph_representation import TensorflowHypergraphRepresentation
from candidate_selection.tensorflow_models.abstract_tensorflow_model import AbstractTensorflowModel
from candidate_selection.tensorflow_models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.tensorflow_models.components.embeddings.sequence_embedding import SequenceEmbedding
from candidate_selection.tensorflow_models.components.embeddings.static_vector_embedding import StaticVectorEmbedding
from candidate_selection.tensorflow_models.components.embeddings.vector_embedding import VectorEmbedding
from candidate_selection.tensorflow_models.components.extras.embedding_retriever import EmbeddingRetriever
from candidate_selection.tensorflow_models.components.extras.mean_gold_embedding_retriever import \
    MeanGoldEmbeddingRetriever
from candidate_selection.tensorflow_models.components.extras.target_comparator import TargetComparator
from candidate_selection.tensorflow_models.components.vector_encoders.multilayer_perceptron import MultilayerPerceptron
from candidate_selection.tensorflow_sentence_representation import TensorflowSentenceRepresentation


class EntityEmbeddingVsGold(AbstractTensorflowModel):


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

        self.mean_gold_embedding_retriever = MeanGoldEmbeddingRetriever(self.variables, variable_prefix="gold_lookup")
        self.add_component(self.mean_gold_embedding_retriever)

        #self.question_sentence = TensorflowSentenceRepresentation(self.variables)
        #self.add_component(self.question_sentence)

        #self.word_embedding = SequenceEmbedding(self.word_indexer, self.variables, variable_prefix="word")
        #self.add_component(self.word_embedding)

        self.target_comparator = TargetComparator(self.variables, variable_prefix="comparison_to_sentence")
        self.add_component(self.target_comparator)

        self.decoder = SoftmaxDecoder(self.variables)
        self.add_component(self.decoder)

        self.sentence_to_graph_mapper = EmbeddingRetriever(self.variables, duplicate_policy="sum", variable_prefix="mapper")
        self.add_component(self.sentence_to_graph_mapper)

        if self.model_settings["use_transformation"]:
            self.transformation = MultilayerPerceptron([self.model_settings["entity_embedding_dimension"],
                                                        self.model_settings["entity_embedding_dimension"]],
                                                       self.variables,
                                                       variable_prefix="transformation",
                                                       l2_scale=self.model_settings["regularization_scale"])

            self.add_component(self.transformation)

    def set_indexers(self, indexers):
        self.entity_indexer = indexers.entity_indexer

    def compute_entity_scores(self):
        self.hypergraph.entity_vertex_embeddings = self.entity_embedding.get_representations()
        gold_embeddings = self.mean_gold_embedding_retriever.get_representations(self.hypergraph.entity_vertex_embeddings)
        #gold_embeddings = tf.Print(gold_embeddings, [gold_embeddings], message="Gold: ", summarize=5)

        if self.model_settings["use_transformation"]:
            gold_embeddings = self.transformation.transform(gold_embeddings)

        #gold_embeddings = tf.Print(gold_embeddings, [self.hypergraph.entity_vertex_embeddings], message="Vertices: ", summarize=100)

        entity_scores = self.target_comparator.get_comparison_scores(gold_embeddings,
                                                                         self.hypergraph.entity_vertex_embeddings)

        #entity_scores = tf.Print(entity_scores, [entity_scores], message="Scores: ", summarize=25)

        return entity_scores