import numpy as np


class VertexFeatureModel:

    features = None
    feature_map = None
    feature_projection = None

    def set_map(self, features, map):
        self.feature_map = map
        self.feature_projection = np.array(list(map.items()))
        self.features = features

    def project(self, keys):
        return np.array([self.feature_map[k] for k in keys])

    def inverse_project(self, features):
        vertices = {feature: [] for feature in features}
        for edge in self.feature_projection:
            if self.feature_map[edge[1]] in features:
                vertices[self.feature_map[edge[1]]].append(edge[0])

        vertices = {n[0]: np.unique(n[1]) for n in vertices.items()}
        print(vertices)
        return vertices
