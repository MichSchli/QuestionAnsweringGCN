import numpy as np


class VertexFeatureModel:

    features = None
    feature_map = None

    def set_map(self, features, map):
        self.feature_map = map
        self.features = features
