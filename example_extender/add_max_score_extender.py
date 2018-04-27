import numpy as np


class AddMaxScoreExtender:

    inner = None

    def __init__(self, inner):
        self.inner = inner

    def extend(self, example):
        example = self.inner.extend(example)

        if not example.has_mentions():
            return example

        for i in range(example.graph.vertices.shape[0]):
            max_score = None
            for j in example.graph.nearby_centroid_map[i]:
                for mention in example.mentions:
                    if mention.entity_index == j and (max_score is None or mention.score > max_score):
                        max_score = mention.score
            example.graph.vertex_max_scores[i] = max_score

        return example