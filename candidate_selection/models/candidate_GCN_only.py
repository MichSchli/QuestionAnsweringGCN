import numpy as np
import tensorflow as tf

from candidate_selection.models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.models.components.graph_encoders.gcn_message_passer import GcnConcatMessagePasser
from candidate_selection.models.components.graph_encoders.hypergraph_gcn_propagation_unit import \
    HypergraphGcnPropagationUnit
from candidate_selection.models.components.graph_encoders.vertex_embedding import VertexEmbedding
from candidate_selection.models.lazy_indexer import LazyIndexer
from candidate_selection.tensorflow_hypergraph_representation import TensorflowHypergraphRepresentation
from candidate_selection.tensorflow_variables_holder import TensorflowVariablesHolder


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

        self.entity_embedding = VertexEmbedding(facts, self.variables, self.dimension,random=False)
        self.event_embedding = VertexEmbedding(facts, self.variables, self.dimension,random=True)

        self.hypergraph = TensorflowHypergraphRepresentation(self.variables)

        self.hypergraph_gcn_propagation_units = [None]*layers
        for layer in range(layers):
            self.hypergraph_gcn_propagation_units[layer] = HypergraphGcnPropagationUnit("layer_"+str(layer), facts, self.variables, dimension, self.hypergraph)

        self.decoder = SoftmaxDecoder(self.variables)

        self.entity_indexer = LazyIndexer()
        self.relation_indexer = LazyIndexer()

    def prepare_variables(self):
        self.hypergraph.prepare_variables()
        self.entity_embedding.prepare_variables()
        self.event_embedding.prepare_variables()
        self.decoder.prepare_variables()

        for hgpu in self.hypergraph_gcn_propagation_units:
            hgpu.prepare_variables()

    def train(self, hypergraph_batch, sentence_batch, gold_prediction_batch):
        pass

    def preprocess(self, hypergraph_batch):
        edges = []

        # Make sender index list
        # Make receiver index list
        # Make type index list

        #vertex_list_slices, vertices = self.preprocess_vertices(hypergraph_batch)

        vertex_list_slices = np.empty((len(hypergraph_batch),2), dtype=np.int32)

        event_to_entity_edges = np.empty((0,2), dtype=np.int32)
        entity_to_event_edges = np.empty((0,2), dtype=np.int32)
        entity_to_entity_edges = np.empty((0,2), dtype=np.int32)
        event_to_entity_types = np.empty((0,), dtype=np.int32)
        entity_to_event_types = np.empty((0,), dtype=np.int32)
        entity_to_entity_types = np.empty((0,), dtype=np.int32)

        entity_map = np.empty(0, dtype=np.int32)

        for i,hypergraph in enumerate(hypergraph_batch):
            phg = self.preprocess_single_hypergraph(hypergraph,
                                                    0 if i == 0 else vertex_list_slices[i-1][0],
                                                    0 if i == 0 else vertex_list_slices[i-1][1])
            vertex_list_slices[i][0] = phg[0]
            vertex_list_slices[i][1] = phg[1]

            if phg[2]:
                event_to_entity_edges = np.concatenate((event_to_entity_edges, phg[2]))
                event_to_entity_types = np.concatenate((event_to_entity_types, phg[3]))

            if phg[4]:
                entity_to_event_edges = np.concatenate((entity_to_event_edges, phg[4]))
                entity_to_event_types = np.concatenate((entity_to_event_types, phg[5]))

            if phg[6]:
                entity_to_entity_edges = np.concatenate((entity_to_entity_edges, phg[6]))
                entity_to_entity_types = np.concatenate((entity_to_entity_types, phg[7]))

            entity_map = np.concatenate((entity_map, phg[8]))

        entity_vertex_slices = vertex_list_slices[:,1]
        entity_vertex_matrix = self.get_padded_vertex_lookup_matrix(entity_vertex_slices, hypergraph_batch)

        n_entities = np.max(entity_vertex_matrix)
        n_events = np.sum(vertex_list_slices[:,0])

        return entity_vertex_matrix, \
               entity_vertex_slices, \
               entity_map, \
               event_to_entity_edges, \
               event_to_entity_types, \
               entity_to_event_edges, \
               entity_to_event_types, \
               entity_to_entity_edges, \
               entity_to_entity_types, \
               n_events

    def get_padded_vertex_lookup_matrix(self, entity_vertex_slices, hypergraph_batch):
        max_vertices = np.max(entity_vertex_slices)
        vertex_matrix = np.zeros((len(hypergraph_batch), max_vertices), dtype=np.int32)
        count = 0
        for i, n in enumerate(entity_vertex_slices):
            vertex_matrix[i][:n] = np.arange(n) + 1 + count
            count += n
        return vertex_matrix

    def preprocess_single_hypergraph(self, hypergraph, event_start_index, entity_start_index):
        event_vertices = hypergraph.get_hypergraph_vertices()
        other_vertices = hypergraph.get_entity_vertices()

        vertex_map = self.entity_indexer.index(other_vertices)

        event_indexes = {k:v+event_start_index for v, k in enumerate(event_vertices)}
        entity_indexes = {k:v+entity_start_index for v, k in enumerate(other_vertices)}

        self.entity_dict.update({v+entity_start_index:k for v, k in enumerate(other_vertices)})

        n_event_vertices = event_vertices.shape[0]
        n_entity_vertices = other_vertices.shape[0]

        event_to_entity_edges = []
        event_to_entity_types = []
        entity_to_event_edges = []
        entity_to_event_types = []
        entity_to_entity_edges = []
        entity_to_entity_types = []

        edges = hypergraph.get_edges()
        edges[:,1] = self.relation_indexer.index(edges[:,1])

        for edge in hypergraph.get_edges():
            if edge[0] in event_vertices and not edge[2] in event_vertices:
                event_to_entity_edges.append([event_indexes[edge[0]], entity_indexes[edge[2]]])
                event_to_entity_types.append(edge[1])
            elif edge[2] in event_vertices and not edge[0] in event_vertices:
                entity_to_event_edges.append([entity_indexes[edge[0]], event_indexes[edge[2]]])
                entity_to_event_types.append(edge[1])
            elif not edge[0] in event_vertices and not edge[2] in event_vertices:
                entity_to_entity_edges.append([entity_indexes[edge[0]], entity_indexes[edge[2]]])
                entity_to_entity_types.append(edge[1])
            else:
                print("Encountered an event to event edge. Shutting down.")
                exit()

        return n_event_vertices, \
               n_entity_vertices, \
               event_to_entity_edges, \
               event_to_entity_types, \
               entity_to_event_edges, \
               entity_to_event_types, \
               entity_to_entity_edges, \
               entity_to_entity_types, \
               vertex_map

    def handle_variable_assignment(self, preprocessed_batch):
        self.decoder.handle_variable_assignment(preprocessed_batch[0], preprocessed_batch[1])
        self.event_embedding.handle_variable_assignment(preprocessed_batch[9])
        self.entity_embedding.handle_variable_assignment(preprocessed_batch[2])
        self.hypergraph.handle_variable_assignment(preprocessed_batch[3:9])

        return self.variables.get_assignment_dict()
        #return [self.variables.vertex_lookup_matrix, self.variables.vertex_count_per_hypergraph, self.variables.number_of_elements_in_batch]

    def retrieve_entities(self, entity_index):
        return [self.entity_dict[entity_index]]

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



