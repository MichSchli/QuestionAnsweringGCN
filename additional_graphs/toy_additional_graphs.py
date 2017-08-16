from model.hypergraph import Hypergraph
import numpy as np


class ToyAdditionalGraphs:

    hypergraphs = None
    mappings = None

    def __init__(self):
        g1 = Hypergraph()
        g1.add_vertices(np.array([["a", "entity"], ["b", "entity"], ["e", "event"]]))
        g1.add_edges(np.array([["a", "r1", "e"], ["a", "r2", "e"]]))
        self.hypergraphs = [g1, g1]
        self.mappings = [{"a": "Alice", "b": "Bob"}, {"a": "BobCorp", "b": "Bob"}]

    def produce_additional_graphs(self):
        for hypergraph, mapping in zip(self.hypergraphs, self.mappings):
            yield hypergraph, mapping