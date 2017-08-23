import numpy as np

from database_interface.data_interface.edge_query_result import EdgeQueryResult
from database_interface.search_filters.prefix_filter import PrefixFilter


class CsvInterface:

    edges = None

    def __init__(self, filename, delimiter=","):
        edges = []
        for line in open(filename):
            edge = line.strip().split(delimiter)
            edges.append(edge)

        self.edges = np.array(edges)

    def get_property(self, vertices, property):
        if vertices.shape[0] == 0:
            return np.array([])

        props = []

        for edge in self.edges:
            if edge[1] == property and edge[0] in vertices:
                props.append([edge[0], edge[2]])

        return np.array(props)

    def get_adjacent_edges(self, node_identifiers, target="entities"):
        edges = EdgeQueryResult()

        target_types = ["entity", "literal"] if target == "entities" else ["event"]

        for edge in self.edges:
            if edge[0] in node_identifiers and edge[4] in target_types:
                edges.append_edge(edge[:3], forward=True)
                edges.append_vertex(edge[2], edge[4])
            elif edge[2] in node_identifiers and edge[3] in target_types:
                edges.append_edge(edge[:3], forward=False)
                edges.append_vertex(edge[0], edge[3])

        return edges