import numpy as np

class IKbInterface:

    def get_neighborhood(self, vertices, edge_limit=None, hops=1):
        #print("Hops: " + str(hops))
        frontier = np.array(vertices)
        seen_edges = []
        seen_vertices = vertices

        while len(seen_edges) < edge_limit and hops > 0 and frontier.shape[0] > 0:
            hops -= 1
            
            remaining_edges = edge_limit-len(seen_edges)

            frontier, edges = self.retrieve_one_neighborhood(frontier, limit=remaining_edges)
            new_vertices = np.isin(frontier, seen_vertices, invert=True)

            #print("Frontier: " + str(frontier.shape))
            frontier = frontier[new_vertices]
            #print("Frontier minus recurrent: " + str(frontier.shape))

            seen_vertices = np.concatenate((seen_vertices, frontier))

            seen_edges.extend(edges)

        return seen_vertices, seen_edges

    def retrieve_one_neighborhood(self, node_identifiers, limit=None):
        return [], []
