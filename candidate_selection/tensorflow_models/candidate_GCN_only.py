import tensorflow as tf
from candidate_selection.models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.models.components.graph_encoders.hypergraph_gcn_propagation_unit import \
    HypergraphGcnPropagationUnit
from candidate_selection.models.components.graph_encoders.vertex_embedding import VertexEmbedding

from candidate_selection.tensorflow_hypergraph_representation import TensorflowHypergraphRepresentation
from candidate_selection.tensorflow_variables_holder import TensorflowVariablesHolder
from experiment_construction.index_construction.indexes.lazy_indexer import LazyIndexer
from input_models.hypergraph.hypergraph_preprocessor import HypergraphPreprocessor


class CandidateGcnOnlyModel:
    """
    A model which computes a max over a GCN representation of the candidate hypergraph.
    """

    dimension = None
    decoder = None
    hypergraph_gcn_propagation_units = None
    facts = None
    variables = None

    embeddings = None

    entity_indexer = None
    relation_indexer = None

    def __init__(self, facts, layers=2, dimension=6):
        self.dimension = dimension
        self.entity_dict = {}
        self.facts = facts
        self.variables = TensorflowVariablesHolder()

        self.hypergraph_batch_preprocessor = HypergraphPreprocessor()

        self.entity_embedding = VertexEmbedding(facts, self.variables, self.dimension,random=False)
        self.event_embedding = VertexEmbedding(facts, self.variables, self.dimension,random=True)

        self.hypergraph = TensorflowHypergraphRepresentation(self.variables)

        self.hypergraph_gcn_propagation_units = [None]*layers
        for layer in range(layers):
            self.hypergraph_gcn_propagation_units[layer] = HypergraphGcnPropagationUnit("layer_"+str(layer), facts, self.variables, dimension, self.hypergraph)

        self.decoder = SoftmaxDecoder(self.variables)

        self.entity_indexer = LazyIndexer()
        self.relation_indexer = LazyIndexer()

    def get_aux_iterators(self):
        return []

    def prepare_variables(self):
        self.hypergraph.prepare_variables()
        self.entity_embedding.prepare_variables()
        self.event_embedding.prepare_variables()
        self.decoder.prepare_variables()

        for hgpu in self.hypergraph_gcn_propagation_units:
            hgpu.prepare_variables()

    def train(self, hypergraph_batch, sentence_batch, gold_prediction_batch):
        pass

    def preprocess(self, batch):
        return self.hypergraph_batch_preprocessor.preprocess(batch[0])

    def handle_variable_assignment(self, preprocessed_batch):
        self.decoder.handle_variable_assignment(preprocessed_batch[0], preprocessed_batch[1])
        self.event_embedding.handle_variable_assignment(preprocessed_batch[9])
        self.entity_embedding.handle_variable_assignment(preprocessed_batch[2])
        self.hypergraph.handle_variable_assignment(preprocessed_batch[3:9])

        return self.variables.get_assignment_dict()

    def retrieve_entities(self, entity_index):
        return [self.hypergraph_batch_preprocessor.retrieve_entity_labels_in_batch(entity_index)]

    def get_prediction_graph(self):
        self.hypergraph.entity_vertex_embeddings = self.entity_embedding.get_representations()
        self.hypergraph.event_vertex_embeddings = self.event_embedding.get_representations()

        for hgpu in self.hypergraph_gcn_propagation_units[:-1]:
            hgpu.propagate()
            self.hypergraph.entity_vertex_embeddings = tf.nn.relu(self.hypergraph.entity_vertex_embeddings)
            self.hypergraph.event_vertex_embeddings = tf.nn.relu(self.hypergraph.event_vertex_embeddings)

        self.hypergraph_gcn_propagation_units[-1].propagate()

        entity_scores = tf.reduce_sum(self.hypergraph.entity_vertex_embeddings,1)
        return self.decoder.decode_to_prediction(entity_scores)



