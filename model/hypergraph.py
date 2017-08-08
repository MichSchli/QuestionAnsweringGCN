import numpy as np


class Hypergraph:

    vertices = None
    edges = None
    vertex_properties = None
    most_recently_added_vertices = None
    unexpanded_vertices = None

    def __init__(self):
        self.vertices = np.array([])
        self.unexpanded_vertices = np.array([])
        self.edges = np.array([]).reshape((0,3))
        self.vertex_properties = {"name": {}, "type": {}}
        self.most_recently_added_vertices = np.array([])

    def pop_unexpanded_hyperedges(self):
        vertex_has_name = np.isin(self.unexpanded_vertices, np.array(list(self.vertex_properties["name"].keys())))
        vertex_is_not_literal = np.array([self.vertex_properties["type"][v] != "literal" for v in self.unexpanded_vertices])
        logically_is_hyperedge = np.logical_and(np.logical_not(vertex_has_name), vertex_is_not_literal)
        unexpanded_hyperedges = self.unexpanded_vertices[logically_is_hyperedge]
        self.unexpanded_vertices = self.unexpanded_vertices[np.logical_not(logically_is_hyperedge)]

        return unexpanded_hyperedges

    def pop_all_unexpanded_vertices(self):
        unexpanded_vertices = self.unexpanded_vertices
        self.unexpanded_vertices = np.array([])
        return unexpanded_vertices

    def get_all_unexpanded_vertices(self):
        return self.unexpanded_vertices

    def add_edges(self, edges):
        self.edges = np.unique(np.vstack((self.edges, np.array(edges))), axis=0)

    def add_vertices(self, vertices):
        vertex_ids = vertices[:,0]
        vertex_types = vertices[:,1]

        unseen_vertices = vertex_ids[np.isin(vertex_ids, self.vertices, invert=True)]
        unseen_vertex_types = vertex_types[np.isin(vertex_ids, self.vertices, invert=True)]

        for v, t in zip(unseen_vertices, unseen_vertex_types):
            self.vertex_properties['type'][v] = t

        self.vertices = np.concatenate((self.vertices, unseen_vertices))
        self.unexpanded_vertices = np.concatenate((self.unexpanded_vertices, unseen_vertices))


    def set_vertex_properties(self, property_dict):
        for k,v in property_dict.items():
            self.set_vertex_property(v, k)

    def set_vertex_property(self, values, property_name):
        if property_name not in self.vertex_properties.keys():
            self.vertex_properties[property_name] = {}

        for row in values:
            self.vertex_properties[property_name][row[0]] = row[1]

    def get_vertices_with_null_property(self, property_name):
        not_present = np.isin(self.vertices, np.array(self.vertex_properties[property_name].keys()), invert=True)
        return self.vertices[not_present]

    def get_most_recently_added_vertices(self):
        return self.most_recently_added_vertices