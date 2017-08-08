import numpy as np


class IKbInterface:

    frontier_filter = None

    def get_neighborhood(self, vertices, edge_limit=None, hops=1):
        frontier = np.array(vertices)
        frontier = self.apply_filter_to_vertices(frontier, self.frontier_filter)

        seen_edges = []
        seen_vertices = vertices

        while len(seen_edges) < edge_limit and hops > 0 and frontier.shape[0] > 0:
            hops -= 1
            
            remaining_edges = edge_limit-len(seen_edges)

            frontier, edges = self.retrieve_one_neighborhood(frontier, limit=remaining_edges)

            new_vertices = np.isin(frontier, seen_vertices, invert=True)
            frontier = frontier[new_vertices]
            frontier = self.apply_filter_to_vertices(frontier, self.frontier_filter)

            seen_vertices = np.concatenate((seen_vertices, frontier))

            seen_edges.extend(edges)

        return seen_vertices, seen_edges

    def apply_filter_to_vertices(self, vertices, filter):
        if filter is None:
            return vertices
        else:
            return vertices[filter.accepts(vertices)]

    """
    Retrieve the one-neighborhood of a vertex according to some filter.
    """
    def retrieve_one_neighborhood(self, node_identifiers, filter=None):
        outgoing_edges, ingoing_edges = self.retrieve_edges(node_identifiers)
        edges = np.concatenate((outgoing_edges, ingoing_edges))

        if outgoing_edges.shape[0] > 0:
            outgoing_entities = self.edge_filter.apply(outgoing_edges)[:,2]
        else:
            outgoing_entities = np.array([])

        if ingoing_edges.shape[0] > 0:
            ingoing_entities = self.edge_filter.apply(ingoing_edges)[:,2]
        else:
            ingoing_entities = np.array([])

        new_entities = np.concatenate((outgoing_entities, ingoing_entities))
        new_entities = self.vertex_frontier_filter.apply(np.unique(new_entities))