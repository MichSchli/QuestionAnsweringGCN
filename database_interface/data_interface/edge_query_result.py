import numpy as np


class EdgeQueryResult:

    forward_edges = None
    backward_edges = None
    vertices = None

    def __init__(self):
        self.forward_edges = []
        self.backward_edges = []
        self.vertices = {}

    def append_edge(self, edge, forward=True):
        if forward:
            self.forward_edges.append(edge)
        else:
            self.backward_edges.append(edge)

    def append_vertex(self, vertex, type):
        self.vertices[vertex] = type

    def get_forward_edges(self):
        return np.array(self.forward_edges)

    def get_backward_edges(self):
        return np.array(self.backward_edges)
