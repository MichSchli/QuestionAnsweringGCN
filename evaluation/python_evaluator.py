import numpy as np

from helpers.read_conll_files import ConllReader


class Evaluation:

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0

    micro_precision = 0
    micro_recall = 0
    micro_f1 = 0

    def __init__(self):
        self.total_true_positives = 0
        self.total_false_positives = 0
        self.total_false_negatives = 0

        self.macro_precision = 0
        self.macro_recall = 0
        self.macro_f1 = 0

        self.micro_precision = 0
        self.micro_recall = 0
        self.micro_f1 = 0

    def pretty_print(self, method="micro"):
        if method == "micro":
            print("Final results (micro-averaged):")
            print("===============================")
            print("Precision: " + str(self.micro_precision))
            print("Recall: " + str(self.micro_recall))
            print("F1: " + str(self.micro_f1))
        else:
            print("Final results (macro-averaged):")
            print("===============================")
            print("Precision: " + str(self.macro_precision))
            print("Recall: " + str(self.macro_recall))
            print("F1: " + str(self.macro_f1))


class Evaluator:

    predictor = None
    gold_reader = None
    evaluation_file = None

    def __init__(self, gold_reader):
        self.gold_reader = gold_reader

    def evaluate(self, model):
        return self.parse_file(model, self.evaluation_file, verbose=False)

    def parse_file(self, prediction_iterator, method="micro", verbose=True):
        gold_iterator = self.gold_reader.iterate()

        evaluation = Evaluation()
        count = 0

        for prediction, gold in zip(prediction_iterator, gold_iterator):
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

            if inner_precision + inner_recall > 0:
                evaluation.macro_f1 += 2 * (inner_precision * inner_recall) / (inner_precision + inner_recall)

        evaluation.micro_precision = evaluation.total_true_positives / (evaluation.total_true_positives + evaluation.total_false_positives)
        evaluation.micro_recall = evaluation.total_true_positives / (evaluation.total_true_positives + evaluation.total_false_negatives)
        evaluation.micro_f1 = 2 * (evaluation.micro_precision * evaluation.micro_recall) / (evaluation.micro_precision + evaluation.micro_recall)

        evaluation.macro_precision /= count
        evaluation.macro_recall /= count
        evaluation.macro_f1 /= count

        if verbose:
            evaluation.pretty_print(method=method)

        return evaluation
