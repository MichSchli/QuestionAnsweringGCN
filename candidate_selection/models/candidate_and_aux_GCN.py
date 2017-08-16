import numpy as np
import tensorflow as tf

from candidate_selection.hypergraph_batch_preprocessor import HypergraphBatchPreprocessor
from candidate_selection.models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.models.components.extras.embedding_mapper import EmbeddingMapper
from candidate_selection.models.components.graph_encoders.gcn_message_passer import GcnConcatMessagePasser
from candidate_selection.models.components.graph_encoders.hypergraph_gcn_propagation_unit import \
    HypergraphGcnPropagationUnit
from candidate_selection.models.components.graph_encoders.vertex_embedding import VertexEmbedding
from candidate_selection.models.lazy_indexer import LazyIndexer
from candidate_selection.tensorflow_hypergraph_representation import TensorflowHypergraphRepresentation
from candidate_selection.tensorflow_variables_holder import TensorflowVariablesHolder


class CandidateAndAuxGcnModel:
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

    aux_iterator = None

    def __init__(self, facts, aux_iterator, layers=2, dimension=6):
        self.dimension = dimension
        self.entity_dict = {}
        self.facts = facts

        # TODO this is the most retarded thing
        # #AlternativeFacts #Yolo
        aux_facts = facts

        self.aux_iterator = aux_iterator
        self.variables = TensorflowVariablesHolder()

        self.aux_mapper = EmbeddingMapper(self.variables)

        self.hypergraph_batch_preprocessor = HypergraphBatchPreprocessor()
        self.aux_hypergraph_batch_preprocessor = HypergraphBatchPreprocessor()

        self.entity_embedding = VertexEmbedding(facts, self.variables, self.dimension,random=False)
        self.event_embedding = VertexEmbedding(facts, self.variables, self.dimension,random=True)

        self.aux_entity_embedding = VertexEmbedding(aux_facts, self.variables, self.dimension, random=False, variable_prefix="aux")
        self.aux_event_embedding = VertexEmbedding(aux_facts, self.variables, self.dimension, random=True, variable_prefix="aux")

        self.hypergraph = TensorflowHypergraphRepresentation(self.variables)
        self.aux_hypergraph = TensorflowHypergraphRepresentation(self.variables, variable_prefix="aux")

        self.hypergraph_gcn_propagation_units = [None]*layers
        self.aux_hypergraph_gcn_propagation_units = [None]*layers
        for layer in range(layers):
            self.hypergraph_gcn_propagation_units[layer] = HypergraphGcnPropagationUnit("layer_"+str(layer), aux_facts, self.variables, dimension, self.hypergraph)
            self.aux_hypergraph_gcn_propagation_units[layer] = HypergraphGcnPropagationUnit("aux_layer_"+str(layer), aux_facts, self.variables, dimension, self.aux_hypergraph)

        self.decoder = SoftmaxDecoder(self.variables)

        self.entity_indexer = LazyIndexer()
        self.relation_indexer = LazyIndexer()

    def get_aux_iterators(self):
        return [self.aux_iterator]

    def prepare_variables(self):
        self.aux_mapper.prepare_variables()
        self.hypergraph.prepare_variables()
        self.entity_embedding.prepare_variables()
        self.event_embedding.prepare_variables()

        self.aux_hypergraph.prepare_variables()
        self.aux_entity_embedding.prepare_variables()
        self.aux_event_embedding.prepare_variables()

        self.decoder.prepare_variables()

        for hgpu in self.hypergraph_gcn_propagation_units:
            hgpu.prepare_variables()

        for hgpu in self.aux_hypergraph_gcn_propagation_units:
            hgpu.prepare_variables()

    def train(self, hypergraph_batch, sentence_batch, gold_prediction_batch):
        pass

    def preprocess(self, batch):
        prp = self.hypergraph_batch_preprocessor.preprocess(batch[0])
        prp_aux = self.aux_hypergraph_batch_preprocessor.preprocess([k[0] for k in batch[1]])

        transform = np.empty((0,2))
        for i,element in enumerate(batch[1]):
            transform = np.concatenate((transform, [[self.aux_hypergraph_batch_preprocessor.retrieve_entity_indexes_in_batch(i,k)
                         ,self.hypergraph_batch_preprocessor.retrieve_entity_indexes_in_batch(i,v)] for k,v in element[1].items()]))

        return [prp, prp_aux, transform]

    def handle_variable_assignment(self, o_preprocessed_batch):
        preprocessed_batch = o_preprocessed_batch[0]
        self.decoder.handle_variable_assignment(preprocessed_batch[0], preprocessed_batch[1])
        self.event_embedding.handle_variable_assignment(preprocessed_batch[9])
        self.entity_embedding.handle_variable_assignment(preprocessed_batch[2])
        self.hypergraph.handle_variable_assignment(preprocessed_batch[3:9])

        a_preprocessed_batch = o_preprocessed_batch[1]
        self.aux_event_embedding.handle_variable_assignment(a_preprocessed_batch[9])
        self.aux_entity_embedding.handle_variable_assignment(a_preprocessed_batch[2])
        self.aux_hypergraph.handle_variable_assignment(a_preprocessed_batch[3:9])

        self.aux_mapper.handle_variable_assignment(o_preprocessed_batch[2][:,0], o_preprocessed_batch[2][:,1], a_preprocessed_batch[10], preprocessed_batch[10])

        return self.variables.get_assignment_dict()

    def retrieve_entities(self, entity_index):
        return [self.hypergraph_batch_preprocessor.retrieve_entity_labels_in_batch(entity_index)]

    def get_prediction_graph(self):
        self.hypergraph.entity_vertex_embeddings = self.entity_embedding.get_representations()
        self.hypergraph.event_vertex_embeddings = self.event_embedding.get_representations()

        self.aux_hypergraph.entity_vertex_embeddings = self.aux_entity_embedding.get_representations()
        self.aux_hypergraph.event_vertex_embeddings = self.aux_event_embedding.get_representations()

        for hgpu, a_hgpu in zip(self.hypergraph_gcn_propagation_units[:-1], self.aux_hypergraph_gcn_propagation_units[:-1]):
            a_hgpu.propagate()
            self.aux_hypergraph.entity_vertex_embeddings = tf.nn.relu(self.aux_hypergraph.entity_vertex_embeddings)
            self.aux_hypergraph.event_vertex_embeddings = tf.nn.relu(self.aux_hypergraph.event_vertex_embeddings)

            self.hypergraph.entity_vertex_embeddings += self.aux_mapper.apply_map(self.aux_hypergraph.entity_vertex_embeddings)

            hgpu.propagate()
            self.hypergraph.entity_vertex_embeddings = tf.nn.relu(self.hypergraph.entity_vertex_embeddings)
            self.hypergraph.event_vertex_embeddings = tf.nn.relu(self.hypergraph.event_vertex_embeddings)

        self.aux_hypergraph_gcn_propagation_units[-1].propagate()
        self.hypergraph.entity_vertex_embeddings += self.aux_mapper.apply_map(self.aux_hypergraph.entity_vertex_embeddings)
        self.hypergraph_gcn_propagation_units[-1].propagate()

        entity_scores = tf.reduce_sum(self.hypergraph.entity_vertex_embeddings,1)
        return self.decoder.decode_to_prediction(entity_scores)



