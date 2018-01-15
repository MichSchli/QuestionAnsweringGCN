from experiment_construction.example_processor_construction.abstract_example_processor import AbstractExampleProcessor
import numpy as np

from model.services.split_graph_service import SplitGraphService
from model.services.subsample_vertices_service import SubsampleVerticesService


class SubsampleVerticesExampleProcessor(AbstractExampleProcessor):

    def process_example(self, example, mode="train"):
        service = SubsampleVerticesService(10)
        graph = example["neighborhood"]
        golds = example["gold_entities"]

        new_graph = service.subsample_vertices(graph, golds)

        print(graph)
        print(golds)
        print(new_graph)
        exit()

        if len(example["sentence_entity_map"]) == 0:
            return True

        centroids = [example["neighborhood"].to_index(c) for c in example["sentence_entity_map"][:, 2]]
        centroids = np.concatenate(centroids)
        example["neighborhood"].set_centroids(centroids)
        example["neighborhood"] = split_graph_service.split_graph(graph)

        return True