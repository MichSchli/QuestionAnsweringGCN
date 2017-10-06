import numpy as np


class Evaluator:

    predictor = None
    gold_reader = None

    def __init__(self, predictor, gold_reader):
        self.predictor = predictor
        self.gold_reader = gold_reader

    def parse_file(self, filename, method="micro"):
        prediction_iterator = self.predictor.parse_file(filename)
        gold_iterator = self.gold_reader.parse_file(filename)

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        macro_precision = 0
        macro_recall = 0
        macro_f1 = 0

        count = 0

        for prediction, gold in zip(prediction_iterator, gold_iterator):
            print("Prediction: " + ",".join(prediction))
            print("Gold: " + ",".join(gold))
            print("")
            count += 1
            true_positives = np.isin(prediction, gold)
            false_positives = np.logical_not(true_positives)
            false_negatives = np.isin(gold, prediction, invert=True)

            total_true_positives += np.sum(true_positives)
            total_false_positives += np.sum(false_positives)
            total_false_negatives += np.sum(false_negatives)

            if np.sum(true_positives) + np.sum(false_positives) == 0:
                inner_precision = 1.0
            else:
                inner_precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))

            inner_recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
            macro_precision += inner_precision
            macro_recall += inner_recall

            if inner_precision + inner_recall > 0:
                macro_f1 += 2 * (inner_precision * inner_recall) / (inner_precision + inner_recall)

        precision = total_true_positives / (total_true_positives + total_false_positives)
        recall = total_true_positives / (total_true_positives + total_false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)

        macro_precision /= count
        macro_recall /= count
        macro_f1 /= count

        if method == "micro":
            print("Final results (micro-averaged):")
            print("===============================")
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F1: " + str(f1))
        else:
            print("Final results (macro-averaged):")
            print("===============================")
            print("Precision: " + str(macro_precision))
            print("Recall: " + str(macro_recall))
            print("F1: " + str(macro_f1))
