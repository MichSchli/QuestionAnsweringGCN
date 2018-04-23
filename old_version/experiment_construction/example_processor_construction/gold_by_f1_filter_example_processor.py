from experiment_construction.example_processor_construction.abstract_example_processor import AbstractExampleProcessor
import numpy as np

class GoldByF1FilterExampleProcessor(AbstractExampleProcessor):

    def process_example(self, example, mode="train"):
        if mode != "train":
            return True

        true_golds = example["true_gold"]
        train_golds = example["gold_entities"]

        centroid_dict = {}

        for gold in train_golds:
            associated_centroids = example["neighborhood"].get_nearby_centroids(gold)
            for centroid in associated_centroids:
                if centroid not in centroid_dict:
                    centroid_dict[centroid] = []
                centroid_dict[centroid].append(gold)

        best_centroid = None
        best_f1 = -1

        for centroid,centroid_golds in centroid_dict.items():
            prediction = [example["neighborhood"].from_index_with_names(cg) for cg in centroid_golds]
            prediction = np.unique(prediction)
            f1 = self.calculate_f1(prediction, true_golds)
            if f1 > best_f1:
                best_f1 = f1
                best_centroid = centroid

        if best_centroid is None:
            return False

        example["gold_entities"] = centroid_dict[best_centroid]

        return True

    def calculate_f1(self, prediction, gold):
        true_positives = np.isin(prediction, gold)
        false_positives = np.logical_not(true_positives)
        false_negatives = np.isin(gold, prediction, invert=True)

        if np.sum(true_positives) + np.sum(false_positives) == 0:
            precision = 1.0
        else:
            precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))

        recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))

        return 2 * (precision * recall) / (precision + recall)