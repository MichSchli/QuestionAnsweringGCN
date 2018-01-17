from experiment_construction.example_processor_construction.abstract_example_processor import AbstractExampleProcessor
import numpy as np

from model.services.split_graph_service import SplitGraphService
from model.services.subsample_vertices_service import SubsampleVerticesService


class SubsampleVerticesExampleProcessor(AbstractExampleProcessor):

    def process_example(self, example, mode="train"):
        if mode != "train":
            return True

        service = SubsampleVerticesService(10)
        graph = example["neighborhood"]
        gold_indexes = example["gold_entities"]

        new_graph, new_golds = service.subsample_vertices(graph, gold_indexes)
        example["neighborhood"] = new_graph
        example["gold_entities"] = new_golds

        return True