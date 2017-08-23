from model.hypergraph import Hypergraph
import numpy as np

from model.hypergraph_model import HypergraphModel


class ToyAdditionalGraphs:

    hypergraphs = None
    mappings = None

    def __init__(self):
        g1 = HypergraphModel()
        g1.add_vertices(np.array(["a", "b"]), type="entities")
        g1.add_vertices(np.array(["e"]), type="events")
        g1.populate_discovered(type="entities")
        g1.populate_discovered(type="events")
        g1.append_edges(np.array([["a", "r1", "e"], ["b", "r2", "e"]]), sources="entities", targets="events")

        self.hypergraphs = [g1, g1]
        self.mappings = [{"a": "http://rdf.freebase.com/ns/Alice", "b": "http://rdf.freebase.com/ns/Bob"},
                         {"a": "http://rdf.freebase.com/ns/BobCorp", "b": "http://rdf.freebase.com/ns/Bob"}]

    def produce_additional_graphs(self):
        for hypergraph, mapping in zip(self.hypergraphs, self.mappings):
            yield hypergraph, mapping