from model.hypergraph_model import HypergraphModel
import numpy as np


class SubsampleVerticesService:

    negative_sample_rate = None

    def __init__(self, negative_sample_rate):
        self.negative_sample_rate = negative_sample_rate

    def subsample_vertices(self, graph, positives):
        new_graph = HypergraphModel()
        new_graph.name_edge_type = graph.name_edge_type
        new_graph.type_edge_type = graph.type_edge_type
        new_graph.relation_map = graph.relation_map
        new_graph.entity_map = {}
        new_graph.inverse_entity_map = {}

        centroids = graph.centroids
        potential_negatives = np.random.randint(0,graph.entity_vertices.shape[0], size=self.negative_sample_rate)

        kept_vertices = np.unique(np.concatenate((positives, centroids, potential_negatives)))

        new_centroids = np.array([np.squeeze(np.argwhere(kept_vertices == e)) for e in centroids])
        new_golds = np.array([np.squeeze(np.argwhere(kept_vertices == e)) for e in positives])

        new_graph.centroids = new_centroids

        new_graph.entity_vertices = graph.entity_vertices[kept_vertices]

        entity_to_entity, entity_to_event, event_to_entity, events_to_keep = graph.get_paths_to_neighboring_centroid_formal_todo_rename(kept_vertices)
        new_graph.event_vertices = np.arange(events_to_keep.shape[0])

        entity_map = {idx:k for k,idx in enumerate(kept_vertices)}
        event_map = {idx:k for k,idx in enumerate(events_to_keep)}

        for i in range(entity_to_entity.shape[0]):
            entity_to_entity[i][0] = entity_map[entity_to_entity[i][0]]
            entity_to_entity[i][2] = entity_map[entity_to_entity[i][2]]

        for i in range(entity_to_event.shape[0]):
            entity_to_event[i][0] = entity_map[entity_to_event[i][0]]
            entity_to_event[i][2] = event_map[entity_to_event[i][2]]

        for i in range(event_to_entity.shape[0]):
            event_to_entity[i][0] = event_map[entity_to_entity[i][0]]
            event_to_entity[i][2] = entity_map[entity_to_entity[i][2]]

        new_graph.entity_to_entity_edges = entity_to_entity
        new_graph.event_to_entity_edges = event_to_entity
        new_graph.entity_to_event_edges = entity_to_event

        names = {}

        for k,v in entity_map.items():
            new_graph.entity_map[v] = graph.entity_map[k]
            if graph.entity_map[k] not in new_graph.inverse_entity_map:
                new_graph.inverse_entity_map[graph.entity_map[k]] = []
            new_graph.inverse_entity_map[graph.entity_map[k]].append(v)

            if graph.has_name(k):
                names[v] = graph.get_name(k)

        new_graph.add_names(names)
        return new_graph, new_golds