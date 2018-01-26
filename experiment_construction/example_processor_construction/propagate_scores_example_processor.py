from experiment_construction.example_processor_construction.abstract_example_processor import AbstractExampleProcessor
import numpy as np


class PropagateScoresExampleProcessor(AbstractExampleProcessor):

    def process_example(self, example, mode="train"):
        if len(example["sentence_entity_map"]) == 0:
            example["neighborhood"].set_scores_to_zero()
            return mode != "train"

        scores = example["sentence_entity_map"][:, 3].astype(np.float32)
        example["neighborhood"].propagate_scores(scores)

        return True