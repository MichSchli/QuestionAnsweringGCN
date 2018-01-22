from experiment_construction.example_processor_construction.abstract_example_processor import AbstractExampleProcessor
import numpy as np

from model.services.split_graph_service import SplitGraphService
from model.services.simple_split_graph_service import SimpleSplitGraphService


class SimpleSplitExampleProcessor(AbstractExampleProcessor):

    def process_example(self, example, mode="train"):
        split_graph_service = SimpleSplitGraphService()
        graph = example["neighborhood"]

        if len(example["sentence_entity_map"]) == 0:
            return True

        centroids = [example["neighborhood"].to_index(c) for c in example["sentence_entity_map"][:, 2]]
        centroids = np.concatenate(centroids)
        example["neighborhood"].set_centroids(centroids)
        example["neighborhood"] = split_graph_service.split_graph(graph)

        return True