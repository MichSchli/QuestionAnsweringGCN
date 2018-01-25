from experiment_construction.example_processor_construction.abstract_example_processor import AbstractExampleProcessor
import numpy as np

from model.services.split_graph_service import SplitGraphService


class GraphSplitExampleProcessor(AbstractExampleProcessor):

    def process_example(self, example, mode="train"):
        split_graph_service = SplitGraphService()
        graph = example["neighborhood"]

        if len(example["sentence_entity_map"]) == 0:
            return True

        new_graph, new_golds = split_graph_service.split_graph(graph, example["gold_entities"])

        example["neighborhood"] = new_graph
        example["gold_entities"] = new_golds

        return True