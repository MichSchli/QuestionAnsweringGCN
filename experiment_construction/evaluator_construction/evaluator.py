from experiment_construction.evaluator_construction.evaluation import Evaluation
import numpy as np


class Evaluator:

    gold_reader = None
    logger = None
    method = None

    def __init__(self, gold_reader, cutoff, logger, method="macro"):
        self.gold_reader = gold_reader
        self.logger = logger
        self.method = method
        self.cutoff = cutoff

    def evaluate(self, prediction_iterator):
        gold_iterator = self.gold_reader.iterate()

        evaluation = Evaluation(self.method)
        count = 0

        for prediction, gold in zip(prediction_iterator, gold_iterator):
            prediction = self.apply_cutoff(prediction)

            print(prediction[:min(5, len(prediction))])
            print(gold["gold_entities"])

            prediction = [p[0] for p in prediction]
            gold = gold["gold_entities"]
            count += 1
            true_positives = np.isin(prediction, gold)
            false_positives = np.logical_not(true_positives)
            false_negatives = np.isin(gold, prediction, invert=True)

            evaluation.total_true_positives += np.sum(true_positives)
            evaluation.total_false_positives += np.sum(false_positives)
            evaluation.total_false_negatives += np.sum(false_negatives)

            if np.sum(true_positives) + np.sum(false_positives) == 0:
                inner_precision = 1.0
            else:
                inner_precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))

            inner_recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
            evaluation.macro_precision += inner_precision
            evaluation.macro_recall += inner_recall

        if np.sum(evaluation.total_true_positives) + np.sum(evaluation.total_false_positives) == 0:
            evaluation.micro_precision = 1.0
        else:
            evaluation.micro_precision = evaluation.total_true_positives / (evaluation.total_true_positives + evaluation.total_false_positives)

        evaluation.micro_recall = evaluation.total_true_positives / (evaluation.total_true_positives + evaluation.total_false_negatives)

        if evaluation.micro_precision + evaluation.micro_recall > 0:
            evaluation.micro_f1 = 2 * (evaluation.micro_precision * evaluation.micro_recall) / (evaluation.micro_precision + evaluation.micro_recall)
        else:
            evaluation.micro_f1 = 0.0

        evaluation.macro_precision /= count
        evaluation.macro_recall /= count

        if evaluation.macro_precision + evaluation.macro_recall > 0:
            evaluation.macro_f1 = 2 * (evaluation.macro_precision * evaluation.macro_recall) / (evaluation.macro_precision + evaluation.macro_recall)
        else:
            evaluation.macro_f1 = 0.0

        evaluation.n_samples = count

        return evaluation

    def apply_cutoff(self, prediction):
        i = 0
        while i < len(prediction) and prediction[i][1] > self.cutoff:
            i += 1
        return prediction[:i]