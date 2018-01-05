from experiment_construction.example_processor_construction.abstract_example_processor import AbstractExampleProcessor
import numpy as np


class GraphSplitExampleProcessor(AbstractExampleProcessor):

    def process_example(self, example, mode="train"):
        graph = example["neighborhood"]
        centroids = [example["neighborhood"].to_index(c) for c in example["sentence_entity_map"][:, 2]]
        centroids = np.concatenate(centroids)
        example["neighborhood"].set_centroids(centroids)
        example["neighborhood"] = graph.get_split_graph()

        return True