import numpy as np


class HypergraphInputModel:

    entity_vertex_matrix = None
    entity_vertex_slices = None
    entity_map = None
    event_to_entity_edges = None
    event_to_entity_types = None
    entity_to_event_edges = None
    entity_to_event_types = None
    entity_to_entity_edges = None
    entity_to_entity_types = None
    n_events = None
    n_entities = None
    in_batch_indices = None
    entity_embeddings = None

    def get(self, tensor_name):
        if tensor_name == "entity_vertex_matrix":
            return self.entity_vertex_matrix

    def retrieve_index_in_batch(self, graph_index, entity_label):
        if entity_label not in self.in_batch_indices[graph_index]:
            print(entity_label)
            #print(self.in_batch_indices)
            return 0

        return self.in_batch_indices[graph_index][entity_label]

    def get_instance_indices(self):
        targets = np.zeros(self.n_entities, dtype=np.int32)
        pointer = 0
        counter = 0
        for r in (self.entity_vertex_matrix != 0).sum(1):
            targets[pointer:pointer + r] = counter
            pointer += r
            counter += 1

        return targets
