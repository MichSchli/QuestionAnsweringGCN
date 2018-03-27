class Graph:

    vertices = None
    edges = None
    vertex_label_to_index_map = None
    vertex_index_to_label_map = None
    entity_vertex_indexes = None

    nearby_centroid_map = None

    def __str__(self):
        return str(self.edges)

    def set_label_to_index_map(self, label_to_index_map):
        self.vertex_label_to_index_map = label_to_index_map
        self.vertex_index_to_label_map = {v: k for k, v in label_to_index_map.items()}

    def map_from_label(self, label):
        if label in self.vertex_label_to_index_map:
            return self.vertex_label_to_index_map[label]
        else:
            return -1

    def map_to_label(self, index):
        if index in self.vertex_index_to_label_map:
            return self.vertex_index_to_label_map[index]
        else:
            return "<unknown>"

    def get_nearby_centroids(self, index):
        return self.nearby_centroid_map[index]

    def count_vertices(self):
        return self.vertices.shape[0]

    def get_entity_vertices(self):
        return self.entity_vertex_indexes
