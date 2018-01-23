from model.hypergraph_model import HypergraphModel
import numpy as np


class SimpleSplitGraphService:

    def split_graph(self, graph):
        new_graph = HypergraphModel()
        new_graph.name_edge_type = graph.name_edge_type
        new_graph.type_edge_type = graph.type_edge_type
        new_graph.relation_map = graph.relation_map

        centroids = graph.centroids

        new_e_to_e = []

        for centroid in centroids:
            new_e_to_e.append(centroid)
        centroid_map = {c:i for i,c in enumerate(new_e_to_e)}


        # We need eveyr single centroid-entity edge, so it is safe to add all of them:
        fixed_edges = []
        for edge in graph.entity_to_entity_edges:
            if edge[0] in centroids:
                new_e_to_e.append(edge[2])
                fixed_edge = [centroid_map[edge[0]], edge[1], len(new_e_to_e)-1]
                fixed_edges.append(fixed_edge)
            if edge[2] in centroids:
                new_e_to_e.append(edge[0])
                fixed_edge = [len(new_e_to_e)-1, edge[1], centroid_map[edge[2]]]
                fixed_edges.append(fixed_edge)

        # We need every single centroi-event edge, so all those and all events can be added:

        # Add all ev-en edges without a centroid, splitting ens

        return new_graph