class Graph:

    vertices = None
    edges = None
    vertex_label_to_index_map = None

    nearby_centroid_map = None

    def __str__(self):
        return str(self.edges)

    def map_from_label(self, label):
        return self.vertex_label_to_index_map[label]

    def get_nearby_centroids(self, index):
        return self.nearby_centroid_map[index]

    def count_vertices(self):
        return self.vertices.shape[0]