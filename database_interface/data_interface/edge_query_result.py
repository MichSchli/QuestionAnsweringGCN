import numpy as np


class EdgeQueryResult:

    forward_edges = None
    backward_edges = None
    vertices = None

    finalized_forward = None
    finalized_backward = None

    def __init__(self):
        self.forward_edges = []
        self.backward_edges = []
        self.vertices = {}
        self.finalized_forward = False
        self.finalized_backward = False

    def append_edge(self, edge, forward=True):
        #print(edge)
        if forward:
            self.forward_edges.append(edge)
        else:
            self.backward_edges.append(edge)

    def append_vertex(self, vertex, type):
        self.vertices[vertex] = type

    def get_forward_edges(self):
        #for edge in self.forward_edges:
        #    print(edge)

        # Avoid allocating more arrays than strictly necessary:
        if not self.finalized_forward:
            self.forward_edges = np.array(self.forward_edges)
            self.finalized_forward = True

        return self.forward_edges

    def get_backward_edges(self):
        if not self.finalized_backward:
            self.backward_edges = np.array(self.backward_edges)
            self.finalized_backward = True

        return self.backward_edges

    def __str__(self):
        return "Forward: " + str(self.get_forward_edges()) +\
               " Backward: " + str(self.get_backward_edges())
