import numpy as np
import copy


class Graph:

    vertices = None
    edges = None

    vertex_label_to_index_map = None
    vertex_index_to_label_map = None

    vertex_name_to_index_map = None
    vertex_index_to_name_map = None

    entity_vertex_indexes = None

    nearby_centroid_map = None
    padded_edge_bow_matrix = None

    vertex_types = None

    def copy(self):
        graph = Graph()

        graph.vertices = np.copy(self.vertices)
        graph.edges = np.copy(self.edges)
        graph.entity_vertex_indexes = np.copy(self.entity_vertex_indexes)

        graph.vertex_label_to_index_map = copy.deepcopy(self.vertex_label_to_index_map)
        graph.vertex_index_to_label_map = copy.deepcopy(self.vertex_index_to_label_map)

        graph.vertex_name_to_index_map = copy.deepcopy(self.vertex_name_to_index_map)
        graph.vertex_index_to_name_map = copy.deepcopy(self.vertex_index_to_name_map)

        graph.nearby_centroid_map = copy.deepcopy(self.nearby_centroid_map)
        graph.padded_edge_bow_matrix = np.copy(self.padded_edge_bow_matrix)

        graph.vertex_types = np.copy(self.vertex_types)

        return graph

    def __str__(self):
        return str(self.edges)

    def set_label_to_index_map(self, label_to_index_map):
        self.vertex_label_to_index_map = label_to_index_map
        self.vertex_index_to_label_map = {v: k for k, v in label_to_index_map.items()}

    def set_index_to_name_map(self, index_to_name_map):
        self.vertex_index_to_name_map = index_to_name_map
        self.vertex_name_to_index_map = {}

        for index,name in index_to_name_map.items():
            if name not in self.vertex_name_to_index_map:
                self.vertex_name_to_index_map[name] = [index]
            else:
                self.vertex_name_to_index_map[name].append(index)

    def map_name_indexes(self, label_to_index_map):
        new_vertex_index_to_name_map = {}
        for k,v in self.vertex_index_to_name_map.items():
            if k in label_to_index_map:
                new_vertex_index_to_name_map[label_to_index_map[k]] = v

        self.set_index_to_name_map(new_vertex_index_to_name_map)


    def map_from_label(self, label):
        if label in self.vertex_label_to_index_map:
            return self.vertex_label_to_index_map[label]
        else:
            return -1

    def map_to_label(self, index):
        if index in self.vertex_index_to_label_map:
            return self.vertex_index_to_label_map[index]
        else:
            return "<unknown>"

    def map_from_name_or_label(self, name_or_label):
        if name_or_label in self.vertex_name_to_index_map:
            return self.vertex_name_to_index_map[name_or_label]
        else:
            return [self.map_from_label(name_or_label)]

    def map_to_name_or_label(self, index):
        if index in self.vertex_index_to_name_map:
            return self.vertex_index_to_name_map[index]
        else:
            return self.map_to_label(index)

    def get_nearby_centroids(self, index):
        return self.nearby_centroid_map[index]

    def count_vertices(self):
        return self.vertices.shape[0]

    def get_entity_vertices(self):
        return self.entity_vertex_indexes

    def add_edges(self, edges, new_edge_bow_features=None):
        self.edges = np.concatenate((self.edges, edges))

        if new_edge_bow_features is None:
            new_edge_bow_features = np.zeros((edges.shape[0],
                                              self.padded_edge_bow_matrix.shape[1]),
                                             dtype=np.int32)

        self.padded_edge_bow_matrix = np.concatenate((self.padded_edge_bow_matrix,
                                                      new_edge_bow_features))

    def add_vertices(self, vertices, vertex_types):
        self.vertices = np.concatenate((self.vertices, vertices))
        self.vertex_types = np.concatenate((self.vertex_types, vertex_types))

    def get_vertex_types(self):
        return self.vertex_types