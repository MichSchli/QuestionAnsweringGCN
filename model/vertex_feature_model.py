import numpy as np


class VertexFeatureModel:

    features = None
    feature_map = None

    def set_map(self, map):
        self.feature_map = np.empty_like(map)
        self.features = np.unique(map[:,1])
