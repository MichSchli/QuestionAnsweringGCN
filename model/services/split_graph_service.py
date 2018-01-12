from model.hypergraph_model import HypergraphModel
import numpy as np


class SplitGraphService:

    def split_graph(self, graph):
        new_graph = HypergraphModel()
        new_graph.name_edge_type = graph.name_edge_type
        new_graph.type_edge_type = graph.type_edge_type
        new_graph.relation_map = graph.relation_map

        new_graph.entity_to_entity_edges = []
        new_graph.entity_to_event_edges = []
        new_graph.event_to_entity_edges = []
        new_graph.centroids = []
        v_counter = 0
        e_counter = 0
        new_graph.entity_map = {}
        new_graph.inverse_entity_map = {}

        new_graph.entity_vertices = []
        new_graph.event_vertices = []

        for centroid in graph.centroids:
            in_centroid_entity_map = {centroid: v_counter}
            new_graph.centroids.append(v_counter)
            new_graph.entity_vertices.append(graph.entity_vertices[centroid])
            v_counter += 1
            for edge in graph.entity_to_entity_edges:
                if edge[0] == centroid or edge[2] == centroid:
                    if edge[0] not in in_centroid_entity_map:
                        in_centroid_entity_map[edge[0]] = v_counter
                        new_graph.entity_vertices.append(graph.entity_vertices[edge[0]])
                        v_counter += 1
                    if edge[2] not in in_centroid_entity_map:
                        in_centroid_entity_map[edge[2]] = v_counter
                        new_graph.entity_vertices.append(graph.entity_vertices[edge[2]])
                        v_counter += 1
                    new_graph.entity_to_entity_edges.append([in_centroid_entity_map[edge[0]], edge[1], in_centroid_entity_map[edge[2]]])

            in_centroid_event_map = {}
            for edge in graph.entity_to_event_edges:
                if edge[0] == centroid:
                    if edge[2] not in in_centroid_event_map:
                        in_centroid_event_map[edge[2]] = e_counter
                        new_graph.event_vertices.append(graph.event_vertices[edge[2]])
                        e_counter += 1
                    new_graph.entity_to_event_edges.append(
                        [in_centroid_entity_map[edge[0]], edge[1], in_centroid_event_map[edge[2]]])

            for edge in graph.event_to_entity_edges:
                if edge[2] == centroid:
                    if edge[0] not in in_centroid_event_map:
                        in_centroid_event_map[edge[0]] = e_counter
                        new_graph.event_vertices.append(graph.event_vertices[edge[0]])
                        e_counter += 1
                    new_graph.event_to_entity_edges.append(
                        [in_centroid_event_map[edge[0]], edge[1], in_centroid_entity_map[edge[2]]])

            for edge in graph.entity_to_event_edges:
                if edge[0] != centroid and edge[2] in in_centroid_event_map:
                    if edge[0] not in in_centroid_entity_map:
                        in_centroid_entity_map[edge[0]] = v_counter
                        new_graph.entity_vertices.append(graph.entity_vertices[edge[0]])
                        v_counter += 1
                    new_graph.entity_to_event_edges.append(
                        [in_centroid_entity_map[edge[0]], edge[1], in_centroid_event_map[edge[2]]])

            for edge in graph.event_to_entity_edges:
                if edge[2] != centroid and edge[0] in in_centroid_event_map:
                    if edge[2] not in in_centroid_entity_map:
                        in_centroid_entity_map[edge[2]] = v_counter
                        new_graph.entity_vertices.append(graph.entity_vertices[edge[2]])
                        v_counter += 1
                    new_graph.event_to_entity_edges.append(
                        [in_centroid_event_map[edge[0]], edge[1], in_centroid_entity_map[edge[2]]])

            names = {}
            for k,v in in_centroid_entity_map.items():
                new_graph.entity_map[v] = graph.entity_map[k]
                if graph.entity_map[k] not in new_graph.inverse_entity_map:
                    new_graph.inverse_entity_map[graph.entity_map[k]] = []
                new_graph.inverse_entity_map[graph.entity_map[k]].append(v)

                if graph.has_name(k):
                    names[v] = graph.get_name(k)

            new_graph.add_names(names)

        new_graph.centroids = np.array(new_graph.centroids, dtype=np.int32)
        new_graph.entity_vertices = np.array(new_graph.entity_vertices, dtype=np.int32)
        new_graph.event_vertices = np.array(new_graph.event_vertices, dtype=np.int32)

        if len(new_graph.entity_to_entity_edges) > 0:
            new_graph.entity_to_entity_edges = np.array(new_graph.entity_to_entity_edges, dtype=np.int32)
        else:
            new_graph.entity_to_entity_edges = np.empty((0,3), dtype=np.int32)

        if len(new_graph.entity_to_event_edges) > 0:
            new_graph.entity_to_event_edges = np.array(new_graph.entity_to_event_edges, dtype=np.int32)
        else:
            new_graph.entity_to_event_edges = np.empty((0,3), dtype=np.int32)

        if len(new_graph.event_to_entity_edges) > 0:
            new_graph.event_to_entity_edges = np.array(new_graph.event_to_entity_edges, dtype=np.int32)
        else:
            new_graph.event_to_entity_edges = np.empty((0,3), dtype=np.int32)

        return new_graph