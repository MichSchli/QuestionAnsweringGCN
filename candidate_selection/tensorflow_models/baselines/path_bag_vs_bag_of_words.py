import tensorflow as tf

from candidate_selection.tensorflow_hypergraph_representation import TensorflowHypergraphRepresentation
from candidate_selection.tensorflow_models.abstract_tensorflow_model import AbstractTensorflowModel
from candidate_selection.tensorflow_models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.tensorflow_models.components.embeddings.sequence_embedding import SequenceEmbedding
from candidate_selection.tensorflow_models.components.embeddings.static_vector_embedding import StaticVectorEmbedding
from candidate_selection.tensorflow_models.components.embeddings.vector_embedding import VectorEmbedding
from candidate_selection.tensorflow_models.components.extras.embedding_retriever import EmbeddingRetriever
from candidate_selection.tensorflow_models.components.extras.target_comparator import TargetComparator
from candidate_selection.tensorflow_models.components.graph_encoders.hypergraph_gcn_propagation_unit import \
    HypergraphGcnPropagationUnit
from candidate_selection.tensorflow_models.components.vector_encoders.multilayer_perceptron import MultilayerPerceptron
from experiment_construction.fact_construction.freebase_facts import FreebaseFacts


class PathBagVsBagOfWords(AbstractTensorflowModel):


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

        self.word_embedding = SequenceEmbedding(self.word_indexer, self.variables, variable_prefix="word")
        self.add_component(self.word_embedding)

        self.target_comparator = TargetComparator(self.variables, variable_prefix="comparison_to_sentence", comparison="concat")
        self.add_component(self.target_comparator)

        self.decoder = SoftmaxDecoder(self.variables)
        self.add_component(self.decoder)

        self.hypergraph_gcn_propagation_units = [None] * self.model_settings["n_layers"]
        for layer in range(self.model_settings["n_layers"]):
            self.hypergraph_gcn_propagation_units[layer] = HypergraphGcnPropagationUnit("layer_" + str(layer),
                                                                                        self.facts,
                                                                                        self.variables,
                                                                                        self.model_settings["entity_embedding_dimension"],
                                                                                        self.hypergraph,
                                                                                        weights="identity",
                                                                                        biases="relation_specific",
                                                                                        self_weight="identity",
                                                                                        self_bias="zero",
                                                                                        add_inverse_relations=False)
            self.add_component(self.hypergraph_gcn_propagation_units[layer])

        self.sentence_to_graph_mapper = EmbeddingRetriever(self.variables, duplicate_policy="sum", variable_prefix="mapper")
        self.add_component(self.sentence_to_graph_mapper)

        if False: #self.model_settings["use_transformation"]:
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

        self.final_transformation = MultilayerPerceptron([self.model_settings["word_embedding_dimension"] + self.model_settings["entity_embedding_dimension"],
                                                          4 * self.model_settings["entity_embedding_dimension"],
                                                    1],
                                                   self.variables,
                                                   variable_prefix="transformation",
                                                   l2_scale=self.model_settings["regularization_scale"])
        self.add_component(self.final_transformation)


    def set_indexers(self, indexers):
        self.word_indexer = indexers.word_indexer
        self.relation_indexer = indexers.relation_indexer
        self.entity_indexer = indexers.entity_indexer

    def compute_entity_scores(self):
        #entity_vertex_embeddings = self.entity_embedding.get_representations()
        word_embeddings = self.word_embedding.get_representations()
        #word_embedding_shape = tf.shape(word_embeddings)
        #word_embeddings = tf.reshape(word_embeddings, [-1, self.model_settings["word_embedding_dimension"]])

        #centroid_embeddings = self.sentence_to_graph_mapper.get_forward_embeddings(entity_vertex_embeddings)
        #centroid_embeddings = self.centroid_transformation.transform(centroid_embeddings)
        #word_embeddings += self.sentence_to_graph_mapper.map_backwards(centroid_embeddings)
        #word_embeddings = tf.reshape(word_embeddings, word_embedding_shape)

        self.hypergraph.initialize_zero_embeddings(self.model_settings["entity_embedding_dimension"])
        for hgpu in self.hypergraph_gcn_propagation_units:
            hgpu.propagate()

        entity_scores = self.hypergraph.entity_vertex_embeddings
        bag_of_words = tf.reduce_sum(word_embeddings, 1)

        #if self.model_settings["use_transformation"]:
        #    bag_of_words = self.transformation.transform(bag_of_words)


        hidden = self.target_comparator.get_comparison_scores(bag_of_words, entity_scores)
        entity_scores = tf.squeeze(self.final_transformation.transform(hidden))

        #entity_scores = tf.Print(entity_scores, [entity_scores], summarize=25, message="entity_scores: ")

        #entity_scores = self.target_comparator.get_comparison_scores(bag_of_words,
        #                                                             entity_scores)

        return entity_scores