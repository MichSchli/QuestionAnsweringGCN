import numpy as np


class VertexFeatureModel:

    features = None
    feature_map = None
    feature_projection = None

    def __init__(self):
        self.feature_map = {}
        self.features = []
        self.feature_projection = np.empty((0,2))

    def set_map(self, features, map):
        self.feature_map = map
        self.feature_projection = np.array(list(map.items()))
        self.features = features

    def add_features(self, map):
        print(map)
        exit()

    def has_projection(self, key):
        return key in self.feature_map

    def project_singleton(self, key):
        if key in self.feature_map:
            return self.features[self.feature_map[key]]
        else:
            return "no_map_found"

    def project(self, keys):
        for k in keys:
            if k not in self.feature_map:
                print("RETRIEVAL_ERROR")
        return np.array([self.features[self.feature_map[k]] if k in self.feature_map else "RETRIEVAL_ERROR" for k in keys])

    def inverse_project(self, features):
        vertices = {feature: [] for feature in features}
        for edge in self.feature_projection:
            if self.features[edge[1]] in features:
                vertices[self.features[edge[1]]].append(edge[0])

        vertices = {n[0]: np.unique(n[1]) for n in vertices.items()}
        #print(vertices)
        return vertices
