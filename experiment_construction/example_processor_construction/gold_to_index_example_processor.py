import numpy as np

from experiment_construction.example_processor_construction.abstract_example_processor import AbstractExampleProcessor


class GoldToIndexExampleProcessor(AbstractExampleProcessor):

    def process_example(self, example, mode="train"):
        names = example["gold_entities"]
        graph = example["neighborhood"]
        gold_list = []
        for name in names:
            if graph.has_index(name):
                gold_list.extend(graph.to_index(name))

        gold_list = np.array(gold_list).astype(np.int32)
        example["gold_entities"] = gold_list
        example["true_gold"] = names
        return True
