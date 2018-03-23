import numpy as np


class VertexFeatureModel:

    feature_map = None

    def __init__(self):
        self.feature_map = {}

    def set_map(self, features):
        self.feature_map = features

    def add_features(self, map):
        self.feature_map.update(map)

    def has_projection(self, key):
        return key in self.feature_map

    def project_singleton(self, key):
        if key in self.feature_map:
            return self.feature_map[key]
        else:
            return "no_map_found"

    def project(self, keys):
        for k in keys:
            if k not in self.feature_map:
                print("RETRIEVAL_ERROR")
        return np.array([self.feature_map[k] if k in self.feature_map else "RETRIEVAL_ERROR" for k in keys])

    def inverse_project(self, features):
        vertices = {feature: [] for feature in features}
        for edge in self.feature_map.items():
            if edge[1] in features:
                vertices[edge[1]].append(edge[0])

        vertices = {n[0]: np.unique(n[1]) for n in vertices.items()}
        #print(vertices)
        return vertices
