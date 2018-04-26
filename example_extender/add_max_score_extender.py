import numpy as np


class AddMaxScoreExtender:

    inner = None

    def __init__(self, inner):
        self.inner = inner

    def extend(self, example):
        example = self.inner.extend(example)

        for i in range(example.graph.vertices.shape[0]):
            max_score = max([example.mentions[j].score for j in example.graph.nearby_centroid_map[i]])
            example.graph.vertex_max_scores[i] = max_score

        if not example.has_mentions():
            return example

        return example