import numpy as np


class VertexFeatureModel:

    features = None
    feature_map = None
    feature_projection = None

    def set_map(self, features, map):
        self.feature_map = map
        self.feature_projection = np.array(features.items())
        self.features = features

    def project(self, keys):
        return np.array([self.feature_map[k] for k in keys])
