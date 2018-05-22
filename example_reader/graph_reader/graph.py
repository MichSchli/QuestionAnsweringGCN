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

    vertex_max_scores = None

    edge_types = None

    def copy(self):
        graph = Graph()

        graph.vertices = np.copy(self.vertices)
        graph.vertex_max_scores = np.zeros(self.vertices.shape[0], dtype=np.float32)
        graph.edges = np.copy(self.edges)
        graph.entity_vertex_indexes = np.copy(self.entity_vertex_indexes)
        graph.centroid_indexes = np.copy(self.centroid_indexes)

        graph.general_vertex_to_entity_index_map = copy.deepcopy(self.general_vertex_to_entity_index_map)

        graph.vertex_label_to_index_map = copy.deepcopy(self.vertex_label_to_index_map)
        graph.vertex_index_to_label_map = copy.deepcopy(self.vertex_index_to_label_map)

        graph.vertex_name_to_index_map = copy.deepcopy(self.vertex_name_to_index_map)
        graph.vertex_index_to_name_map = copy.deepcopy(self.vertex_index_to_name_map)

        graph.nearby_centroid_map = copy.deepcopy(self.nearby_centroid_map)
        graph.padded_edge_bow_matrix = np.copy(self.padded_edge_bow_matrix)

        graph.vertex_types = np.copy(self.vertex_types)

        graph.entity_centroid_paths = copy.deepcopy(self.entity_centroid_paths)
        graph.edge_types = copy.deepcopy(self.edge_types)

        return graph

    def summarize(self):
        return str(self.padded_edge_bow_matrix)

    def get_entity_path_to_centroid(self, index, relation_index):
        real_index = self.entity_vertex_indexes[index]
        forward_entity_edges = self.edges[np.where(self.edges[:,0] == real_index)[0]]
        backward_entity_edges = self.edges[np.where(self.edges[:,2] == real_index)[0]]
        forward_entity_edges = np.array([forward_entity_edges[:,0], [relation_index.from_index(i) for i in forward_entity_edges[:,1]], forward_entity_edges[:,2]]).transpose()
        backward_entity_edges = np.array([backward_entity_edges[:,2], [relation_index.from_index(i)+"-reverse" for i in backward_entity_edges[:,1]], backward_entity_edges[:,0]]).transpose()

        entity_edges = np.concatenate((forward_entity_edges, backward_entity_edges))

        direct_centroid_links = np.isin(entity_edges[:, 2], [str(x) for x in self.centroid_indexes])

        output = ["|".join([self.vertices[int(edge[0])], edge[1], self.vertices[int(edge[2])]]) for edge in entity_edges[direct_centroid_links]]

        for edge in self.edges:
            if edge[0] in self.centroid_indexes and str(edge[2]) in entity_edges[:,2]:
                connectors = entity_edges[np.where(entity_edges[:,2] == str(edge[2]))[0]]
                for connector in connectors:
                    connector_string = "|".join([self.vertices[int(connector[0])], connector[1], self.vertices[int(connector[2])]])
                    connector_string += "|" + relation_index.from_index(edge[1]) + "-reverse|" + self.vertices[edge[0]]
                    output.append(connector_string)
            elif edge[2] in self.centroid_indexes and str(edge[0]) in entity_edges[:, 2]:
                connectors = entity_edges[np.where(entity_edges[:,2] == str(edge[0]))[0]]
                for connector in connectors:
                    connector_string = "|".join([self.vertices[int(connector[0])], connector[1], self.vertices[int(connector[2])]])
                    connector_string += "|" + relation_index.from_index(edge[1]) + "|" + self.vertices[edge[2]]
                    output.append(connector_string)

        return output

    def map_general_vertex_to_entity_index(self, general_index):
        return self.general_vertex_to_entity_index_map[general_index]

    def __init__(self):
        self.vertex_max_scores = np.zeros(0, dtype=np.int32)

    def __str__(self):
        return str(self.edges)

    def get_max_scores(self):
        return self.vertex_max_scores

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

    def map_name_indexes(self, index_map):
        new_vertex_index_to_name_map = {}
        for old_index, new_index in index_map.items():
            if old_index in self.vertex_index_to_name_map:
                new_vertex_index_to_name_map[new_index] = self.vertex_index_to_name_map[old_index]

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

    def count_entity_vertices(self):
        return self.entity_vertex_indexes.shape[0]

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

        self.vertex_max_scores = np.concatenate((self.vertex_max_scores, np.zeros(vertices.shape[0], dtype=np.float32)))

    def get_vertex_types(self):
        return self.vertex_types

    def get_sentence_vertex_index(self):
        return self.sentence_vertex_index

    def update_general_vertex_to_entity_index_map(self):
        self.general_vertex_to_entity_index_map = {k:i for i,k in enumerate(self.entity_vertex_indexes)}