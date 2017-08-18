import numpy as np

from candidate_selection.models.lazy_indexer import LazyIndexer


class HypergraphBatchPreprocessor:

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
            phg = self.preprocess_single_hypergraph(hypergraph, event_start_index, entity_start_index)
            vertex_list_slices[i][0] = phg[0]
            vertex_list_slices[i][1] = phg[1]

            event_start_index += phg[0]
            entity_start_index += phg[1]

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
               n_events, \
               n_entities

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

    def retrieve_entity_labels_in_batch(self, entity_index):
        return self.in_batch_labels[entity_index]

    def preprocess_single_hypergraph(self, hypergraph, event_start_index, entity_start_index):
        event_vertices = hypergraph.get_hypergraph_vertices()
        other_vertices = hypergraph.get_entity_vertices()

        vertex_map = self.entity_indexer.index(other_vertices)

        event_indexes = {k:v+event_start_index for v, k in enumerate(event_vertices)}
        entity_indexes = {k:v+entity_start_index for v, k in enumerate(other_vertices)}

        self.in_batch_labels.update({v+entity_start_index:k for v, k in enumerate(other_vertices)})
        self.in_batch_indices[self.graph_counter] = {k:v+entity_start_index for v, k in enumerate(other_vertices)}

        n_event_vertices = event_vertices.shape[0]
        n_entity_vertices = other_vertices.shape[0]

        event_to_entity_edges = []
        event_to_entity_types = []
        entity_to_event_edges = []
        entity_to_event_types = []
        entity_to_entity_edges = []
        entity_to_entity_types = []

        edges = hypergraph.get_edges()
        edge_types = self.relation_indexer.index(edges[:,1])

        for edge, edge_type in zip(hypergraph.get_edges(), edge_types):
            if edge[0] in event_vertices and not edge[2] in event_vertices:
                event_to_entity_edges.append([event_indexes[edge[0]], entity_indexes[edge[2]]])
                event_to_entity_types.append(edge_type)
            elif edge[2] in event_vertices and not edge[0] in event_vertices:
                entity_to_event_edges.append([entity_indexes[edge[0]], event_indexes[edge[2]]])
                entity_to_event_types.append(edge_type)
            elif not edge[0] in event_vertices and not edge[2] in event_vertices:
                entity_to_entity_edges.append([entity_indexes[edge[0]], entity_indexes[edge[2]]])
                entity_to_entity_types.append(edge_type)
            else:
                print("Encountered an event to event edge. Shutting down.")
                exit()

        self.graph_counter += 1

        return n_event_vertices, \
               n_entity_vertices, \
               event_to_entity_edges, \
               event_to_entity_types, \
               entity_to_event_edges, \
               entity_to_event_types, \
               entity_to_entity_edges, \
               entity_to_entity_types, \
               vertex_map