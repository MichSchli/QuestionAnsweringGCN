import numpy as np


class Evaluator:

    predictor = None
    gold_reader = None

    def __init__(self, predictor, gold_reader):
        self.predictor = predictor
        self.gold_reader = gold_reader

    def parse_file(self, filename):
        prediction_iterator = self.predictor.parse_file(filename)
        gold_iterator = self.gold_reader.parse_file(filename)

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        for prediction, gold in zip(prediction_iterator, gold_iterator):
            true_positives = np.isin(prediction, gold)
            false_positives = np.logical_not(true_positives)
            false_negatives = np.isin(gold, prediction, invert=True)

            total_true_positives += np.sum(true_positives)
            total_false_positives += np.sum(false_positives)
            total_false_negatives += np.sum(false_negatives)

        precision = total_true_positives / (total_true_positives + total_false_positives)
        recall = total_true_positives / (total_true_positives + total_false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)

        print("Final results (macro-averaged):")
        print("===============================")
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1: " + str(f1))