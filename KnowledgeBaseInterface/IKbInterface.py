import numpy as np

class IKbInterface:

    def get_neighborhood(self, vertices, edge_limit=None, hops=1):
        frontier = np.array(vertices)
        seen_edges = []
        while len(seen_edges) < edge_limit and hops > 0:
            hops -1
            remaining_edges = edge_limit-len(seen_edges)

            frontier, edges = self.retrieve_one_neighborhood(frontier, limit=remaining_edges)
            seen_edges.extend(edges)

        return seen_edges

    def retrieve_one_neighborhood(self, node_identifiers, limit=None):
        return [], []
