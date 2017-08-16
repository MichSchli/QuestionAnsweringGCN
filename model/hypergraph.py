import numpy as np

from database_interface.search_filters.prefix_filter import PrefixFilter


class Hypergraph:

    vertices = None
    edges = None
    vertex_properties = None
    most_recently_added_vertices = None
    unexpanded_vertices = None
    edge_cache = None

    def __init__(self):
        self.vertices = np.array([])
        self.unexpanded_vertices = np.array([])
        self.edges = np.array([]).reshape((0,3))
        self.vertex_properties = {"name": {}, "type": {}}
        self.most_recently_added_vertices = np.array([])
        self.edge_cache = set([])

    def get_edges(self):
        return self.edges

    def get_hypergraph_vertices(self):
        logically_is_hyperedge = self.find_hypergraph_edges(self.vertices)
        return self.vertices[logically_is_hyperedge]

    #get non hypergraph vertices
    def get_entity_vertices(self):
        logically_is_hyperedge = self.find_hypergraph_edges(self.vertices)
        return self.vertices[np.logical_not(logically_is_hyperedge)]

    def pop_unexpanded_hyperedges(self):
        logically_is_hyperedge = self.find_hypergraph_edges(self.unexpanded_vertices)
        unexpanded_hyperedges = self.unexpanded_vertices[logically_is_hyperedge]
        self.unexpanded_vertices = self.unexpanded_vertices[np.logical_not(logically_is_hyperedge)]

        return unexpanded_hyperedges

    def find_hypergraph_edges(self, entities):
        return np.array([self.vertex_properties["type"][v] == "event" for v in entities])

    def pop_all_unexpanded_vertices(self, include_types=False):
        unexpanded_vertices = self.unexpanded_vertices
        self.unexpanded_vertices = np.array([])

        if include_types:
            types = np.array([self.vertex_properties["type"][v] for v in unexpanded_vertices])
            return unexpanded_vertices, types
        else:
            return unexpanded_vertices

    def get_all_unexpanded_vertices(self, include_types=False):
        if include_types:
            types = np.array([self.vertex_properties["type"][v] for v in self.unexpanded_vertices])
            return self.unexpanded_vertices, types
        else:
            return self.unexpanded_vertices

    def add_edges(self, edges):
        if edges.shape[0] == 0:
            return

        #print(edges.shape)
        novel = np.ones(edges.shape[0], dtype=bool)
        for i,edge in enumerate(edges):
            str_edge = edge[0] + edge[1] + edge[2]
            if str_edge in self.edge_cache:
                novel[i] = False
            else:
                self.edge_cache.add(str_edge)

            #if edge[0] not in self.edge_cache:
            #    self.edge_cache[edge[0]] = {}

            #if edge[1] not in self.edge_cache[edge[0]]:
            #    self.edge_cache[edge[0]][edge[1]] = []

            #if edge[2] not in self.edge_cache[edge[0]][edge[1]]:
            #    self.edge_cache[edge[0]][edge[1]].append(edge[2])
            #else:
            #    novel[i] = False

        #print(edges[novel].shape)
        self.edges = np.vstack((self.edges, edges[novel]))

    def add_vertices(self, vertices):
        if vertices.shape[0] == 0:
            return

        vertex_ids = vertices[:,0]
        vertex_types = vertices[:,1]
        
        new_vertex_ids = np.isin(vertex_ids, self.vertices, invert=True, assume_unique=True)
        unseen_vertices = vertex_ids[new_vertex_ids]
        unseen_vertex_types = vertex_types[new_vertex_ids]

        for v, t in zip(unseen_vertices, unseen_vertex_types):
            self.vertex_properties['type'][v] = t

        self.vertices = np.concatenate((self.vertices, unseen_vertices))
        self.unexpanded_vertices = np.concatenate((self.unexpanded_vertices, unseen_vertices))

    def set_vertex_properties(self, property_dict):
        for k,v in property_dict.items():
            self.set_vertex_property(v, k)

    def set_vertex_property(self, values, property_name):
        if values.shape[0] == 0:
            return

        if property_name not in self.vertex_properties.keys():
            self.vertex_properties[property_name] = {}

        for row in values:
            self.vertex_properties[property_name][row[0]] = row[1]

    def get_vertices_with_null_property(self, property_name):
        not_present = np.isin(self.vertices, np.array(self.vertex_properties[property_name].keys()), invert=True)
        return self.vertices[not_present]

    def get_most_recently_added_vertices(self, include_types=True):
        if include_types:
            types = np.array([self.vertex_properties["type"][v] for v in self.most_recently_added_vertices])
            return self.most_recently_added_vertices, types
        else:
            return self.most_recently_added_vertices
