import tensorflow as tf

from candidate_selection.tensorflow_hypergraph_representation import TensorflowHypergraphRepresentation
from candidate_selection.tensorflow_models.abstract_tensorflow_model import AbstractTensorflowModel
from candidate_selection.tensorflow_models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.tensorflow_models.components.embeddings.sequence_embedding import SequenceEmbedding
from candidate_selection.tensorflow_models.components.embeddings.static_vector_embedding import StaticVectorEmbedding
from candidate_selection.tensorflow_models.components.embeddings.vector_embedding import VectorEmbedding
from candidate_selection.tensorflow_models.components.extras.embedding_retriever import EmbeddingRetriever
from candidate_selection.tensorflow_models.components.extras.target_comparator import TargetComparator
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_factory import GcnFactory
from candidate_selection.tensorflow_models.components.graph_encoders.hypergraph_gcn_propagation_unit import \
    HypergraphGcnPropagationUnit
from candidate_selection.tensorflow_models.components.meta_components.candidate_scoring.neural_network_or_factorization_scorer import \
    NeuralNetworkOrFactorizationScorer
from candidate_selection.tensorflow_models.components.sequence_encoders.attention import Attention
from candidate_selection.tensorflow_models.components.sequence_encoders.bilstm import BiLstm
from candidate_selection.tensorflow_models.components.sequence_encoders.multihead_attention import MultiheadAttention
from candidate_selection.tensorflow_models.components.vector_encoders.multilayer_perceptron import MultilayerPerceptron
from experiment_construction.fact_construction.freebase_facts import FreebaseFacts


class PathBagPlusScoreFeaturesWithTypeGatesVsLstm(AbstractTensorflowModel):


    def get_preprocessor_stack_types(self):
        preprocessor_stack_types = ["hypergraph", "gold", "sentence"]
        if self.model_settings["static_entity_embeddings"]:
            preprocessor_stack_types += ["static_entity_embeddings"]
        return preprocessor_stack_types

    def initialize_graph(self):
        self.hypergraph = TensorflowHypergraphRepresentation(self.variables, edge_dropout_rate=self.model_settings["edge_dropout"])
        self.add_component(self.hypergraph)

        self.lstms = [BiLstm(self.variables, self.model_settings["word_embedding_dimension"]+1, self.model_settings["lstm_hidden_state_dimension"], variable_prefix="lstm_" + str(i)) for i in
                      range(self.model_settings["n_lstms"])]
        for lstm in self.lstms:
            self.add_component(lstm)

        self.word_embedding = SequenceEmbedding(self.word_indexer, self.variables, variable_prefix="word", word_dropout_rate=self.model_settings["word_dropout"], is_static=self.model_settings["static_word_embeddings"])
        self.add_component(self.word_embedding)

        #self.attention = Attention(self.model_settings["word_embedding_dimension"], self.variables, variable_prefix="attention", strategy="constant_query")
        self.attention = MultiheadAttention(self.model_settings["lstm_hidden_state_dimension"], self.variables, attention_heads=self.model_settings["n_attention_heads"],
                                   variable_prefix="attention", strategy="constant_query", attention_dropout=self.model_settings["attention_dropout"])

        self.add_component(self.attention)

        self.target_comparator = TargetComparator(self.variables, variable_prefix="comparison_to_sentence", comparison="concat")
        self.add_component(self.target_comparator)

        self.decoder = SoftmaxDecoder(self.variables, self.model_settings["loss"])
        self.add_component(self.decoder)

        self.candidate_scorer = NeuralNetworkOrFactorizationScorer(self.model_settings, self.variables, variable_prefix="scorer")
        self.add_component(self.candidate_scorer)

        gcn_factory = GcnFactory()
        gcn_settings = {"n_layers": self.model_settings["n_layers"],
                        "embedding_dimension": self.model_settings["entity_embedding_dimension"],
                        "n_relation_types": self.facts.number_of_relation_types}

        self.hypergraph_gcn_propagation_units = gcn_factory.get_gated_gcn(self.hypergraph, self.variables, gcn_settings)
        for layer in self.hypergraph_gcn_propagation_units:
            self.add_component(layer)

        self.sentence_to_graph_mapper = EmbeddingRetriever(self.variables, duplicate_policy="sum", variable_prefix="mapper")
        self.add_component(self.sentence_to_graph_mapper)


        self.final_transformation = MultilayerPerceptron([int(self.model_settings["lstm_hidden_state_dimension"]/2) + self.model_settings["entity_embedding_dimension"],
                                                          self.model_settings["nn_hidden_state_dimension"],
                                                    1],
                                                   self.variables,
                                                   variable_prefix="transformation",
                                                   l2_scale=self.model_settings["regularization_scale"])
        self.add_component(self.final_transformation)


    def set_indexers(self, indexers):
        self.word_indexer = indexers.word_indexer
        self.relation_indexer = indexers.relation_indexer
        self.entity_indexer = indexers.entity_indexer

    edge_gates = None

    def get_edge_gates(self):
        if self.edge_gates is None:
            self.edge_gates = [hpgu.get_edge_gates() for hpgu in self.hypergraph_gcn_propagation_units]
        return self.edge_gates

    def compute_entity_scores(self, mode="train"):
        self.hypergraph.initialize_with_centroid_scores()

        word_embeddings = self.word_embedding.get_representations(mode=mode)
        word_embedding_shape = tf.shape(word_embeddings)
        word_embeddings = tf.reshape(word_embeddings, [-1, self.model_settings["word_embedding_dimension"]])
        centroid_embeddings = self.sentence_to_graph_mapper.get_forward_embeddings(tf.ones([tf.shape(self.hypergraph.entity_vertex_embeddings)[0], 1]))
        word_embeddings = tf.concat([word_embeddings, self.sentence_to_graph_mapper.map_backwards(centroid_embeddings)], axis=1)
        word_embeddings = tf.reshape(word_embeddings, [word_embedding_shape[0],-1,self.model_settings["word_embedding_dimension"]+1])

        for lstm in self.lstms:
            word_embeddings = lstm.transform_sequences(word_embeddings)
        sentence_vector = self.attention.attend(word_embeddings, mode=mode)

        for hgpu in self.hypergraph_gcn_propagation_units:
            hgpu.set_gate_key(sentence_vector)
            hgpu.propagate()
        entity_scores = self.hypergraph.entity_vertex_embeddings

        if self.model_settings["concatenate_scores"]:
            entity_scores = tf.concat([entity_scores, tf.expand_dims(self.hypergraph.get_vertex_scores(),1)], axis=1)

        return self.candidate_scorer.score(sentence_vector, entity_scores, mode=mode)