import numpy as np

from experiment_construction.example_processor_construction.abstract_example_processor import AbstractExampleProcessor


class NameToIndexExampleProcessor(AbstractExampleProcessor):

    def process_example(self, example, mode="train"):
        names = example["gold_entities"]
        graph = example["neighborhood"]
        name_projection_dictionary = graph.get_inverse_name_connections(names)

        gold_list = []
        for name, l in name_projection_dictionary.items():
            if len(l) > 0:
                gold_list.extend(l)
            elif graph.has_index(name):
                gold_list.extend(graph.to_index(name))

        # TODO CHECK SOMEWHERE ELSE
        if len(gold_list) == 0:
            if mode == "predict":
                return True

            return False

        gold_list = np.array(gold_list).astype(np.int32)
        # print(example["neighborhood"].entity_vertices.shape[0])
        # print("projected " + str(example["gold_entities"]) + " to " + str(gold_list))
        example["gold_entities"] = gold_list
        example["true_gold"] = names
        return True
