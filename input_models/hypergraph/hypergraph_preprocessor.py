import numpy as np

from candidate_selection.models.lazy_indexer import LazyIndexer
from input_models.hypergraph.hypergraph_input_model import HypergraphInputModel


class HypergraphPreprocessor:

    entity_indexer = None
    relation_indexer = None
    in_batch_indices = None
    in_batch_labels = None

    graph_counter = None

    def __init__(self):
        self.entity_indexer = LazyIndexer()
        self.relation_indexer = LazyIndexer()
        self.in_batch_indices = {}
        self.in_batch_labels = {}
        self.graph_counter = 0

    def preprocess(self, hypergraph_batch):
        #print("Beginning preprocessing...")
        self.in_batch_indices = {}
        self.in_batch_labels = {}
        self.graph_counter = 0
        vertex_list_slices = np.empty((len(hypergraph_batch),2), dtype=np.int32)

        event_to_entity_edges = np.empty((0,2), dtype=np.int32)
        entity_to_event_edges = np.empty((0,2), dtype=np.int32)
        entity_to_entity_edges = np.empty((0,2), dtype=np.int32)
        event_to_entity_types = np.empty((0,), dtype=np.int32)
        entity_to_event_types = np.empty((0,), dtype=np.int32)
        entity_to_entity_types = np.empty((0,), dtype=np.int32)

        entity_map = np.empty(0, dtype=np.int32)

        event_start_index = 0
        entity_start_index = 0
        for i,hypergraph in enumerate(hypergraph_batch):
            #print("Element started")
            phg = self.preprocess_single_hypergraph(hypergraph, event_start_index, entity_start_index)
            vertex_list_slices[i][0] = phg[0]
            vertex_list_slices[i][1] = phg[1]

            event_start_index += phg[0]
            entity_start_index += phg[1]

            if phg[2].shape[0] > 0:
                event_to_entity_edges = np.concatenate((event_to_entity_edges, phg[2]))
                event_to_entity_types = np.concatenate((event_to_entity_types, phg[3]))

            if phg[4].shape[0] > 0:
                entity_to_event_edges = np.concatenate((entity_to_event_edges, phg[4]))
                entity_to_event_types = np.concatenate((entity_to_event_types, phg[5]))

            if phg[6].shape[0] > 0:
                entity_to_entity_edges = np.concatenate((entity_to_entity_edges, phg[6]))
                entity_to_entity_types = np.concatenate((entity_to_entity_types, phg[7]))

            entity_map = np.concatenate((entity_map, phg[8]))

        entity_vertex_slices = vertex_list_slices[:,1]
        #print("Getting vertex lookup matrix")
        entity_vertex_matrix = self.get_padded_vertex_lookup_matrix(entity_vertex_slices, hypergraph_batch)

        #print(entity_vertex_slices)
        #print(entity_vertex_matrix.shape)

        n_entities = np.max(entity_vertex_matrix)
        n_events = np.sum(vertex_list_slices[:,0])

        input_model = HypergraphInputModel()
        input_model.entity_vertex_matrix = entity_vertex_matrix
        input_model.entity_vertex_slices = entity_vertex_slices
        input_model.entity_map = entity_map
        input_model.event_to_entity_edges = event_to_entity_edges
        input_model.event_to_entity_types = event_to_entity_types
        input_model.entity_to_event_edges = entity_to_event_edges
        input_model.entity_to_event_types = entity_to_event_types
        input_model.entity_to_entity_edges = entity_to_entity_edges
        input_model.entity_to_entity_types = entity_to_entity_types
        input_model.n_events = n_events
        input_model.n_entities = n_entities

        return input_model

    def get_padded_vertex_lookup_matrix(self, entity_vertex_slices, hypergraph_batch):
        max_vertices = np.max(entity_vertex_slices)
        vertex_matrix = np.zeros((len(hypergraph_batch), max_vertices), dtype=np.int32)
        count = 0
        for i, n in enumerate(entity_vertex_slices):
            vertex_matrix[i][:n] = np.arange(n) + 1 + count
            count += n
        return vertex_matrix

    def retrieve_entity_indexes_in_batch(self, graph_index, entity_label):
        return self.in_batch_indices[graph_index][entity_label]

    def retrieve_entity_labels_in_batch(self, graph_index, entity_index):
        return self.in_batch_labels[graph_index][entity_index]

    def preprocess_single_hypergraph(self, hypergraph, event_start_index, entity_start_index):
        #print("Preprocessing hgraph")
        event_vertices = hypergraph.get_vertices(type="events")
        entity_vertices = hypergraph.get_vertices(type="entities")
        #print(event_vertices)
        #print(entity_vertices)

        vertex_map = self.entity_indexer.index(entity_vertices)

        event_indexes = {k:v+event_start_index for v, k in enumerate(event_vertices)}
        entity_indexes = {k:v+entity_start_index for v, k in enumerate(entity_vertices)}

        #print(self.graph_counter)
        #print(entity_vertices[:3])

        self.in_batch_labels[self.graph_counter] = {v:k for v, k in enumerate(entity_vertices)}
        self.in_batch_indices[self.graph_counter] = {k:v+entity_start_index for v, k in enumerate(entity_vertices)}

        n_event_vertices = event_vertices.shape[0]
        n_entity_vertices = entity_vertices.shape[0]

        event_to_entity_edges = hypergraph.get_edges(sources="events", targets="entities")
        event_to_entity_types = self.relation_indexer.index(event_to_entity_edges[:,1])
        entity_to_event_edges = hypergraph.get_edges(sources="entities", targets="events")
        entity_to_event_types = self.relation_indexer.index(entity_to_event_edges[:,1])
        entity_to_entity_edges = hypergraph.get_edges(sources="entities", targets="entities")
        entity_to_entity_types = self.relation_indexer.index(entity_to_entity_edges[:,1])

        ev_to_en_2 = np.empty((event_to_entity_edges.shape[0], 2))
        en_to_ev_2 = np.empty((entity_to_event_edges.shape[0], 2))
        en_to_en_2 = np.empty((entity_to_entity_edges.shape[0], 2))

        for i, edge in enumerate(event_to_entity_edges):
            ev_to_en_2[i][0] = event_indexes[edge[0]]
            ev_to_en_2[i][1] = entity_indexes[edge[2]]

        for i, edge in enumerate(entity_to_event_edges):
            en_to_ev_2[i][0] = entity_indexes[edge[0]]
            en_to_ev_2[i][1] = event_indexes[edge[2]]

        for i, edge in enumerate(entity_to_entity_edges):
            en_to_en_2[i][0] = entity_indexes[edge[0]]
            en_to_en_2[i][1] = entity_indexes[edge[2]]

        #print("Doner")

        self.graph_counter += 1

        return n_event_vertices, \
               n_entity_vertices, \
               ev_to_en_2, \
               event_to_entity_types, \
               en_to_ev_2, \
               entity_to_event_types, \
               en_to_en_2, \
               entity_to_entity_types, \
               vertex_map
