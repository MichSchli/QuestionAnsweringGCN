import numpy as np

from model_tester.evaluator.evaluation import Evaluation


class Evaluator:

    evaluation = None

    def __init__(self):
        pass

    def begin_evaluation(self):
        self.evaluation = Evaluation(default_method="macro")

    def add_prediction(self, example):
        true_labels = example.get_gold_labels()
        predicted_labels = example.get_predicted_labels(0.5)
        print(true_labels)
        print(predicted_labels)

        true_positives = np.isin(predicted_labels, true_labels)
        false_positives = np.logical_not(true_positives)
        false_negatives = np.isin(true_labels, predicted_labels, invert=True)

        self.evaluation.total_true_positives += np.sum(true_positives)
        self.evaluation.total_false_positives += np.sum(false_positives)
        self.evaluation.total_false_negatives += np.sum(false_negatives)

        if np.sum(true_positives) + np.sum(false_positives) == 0:
            inner_precision = 1.0
        else:
            inner_precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))

        inner_recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
        self.evaluation.macro_precision += inner_precision
        self.evaluation.macro_recall += inner_recall

        if inner_precision + inner_recall > 0:
            self.evaluation.macro_f1 += 2 * (inner_precision * inner_recall) / (inner_precision + inner_recall)

        self.evaluation.n_samples += 1

    def final_scores(self):
        if np.sum(self.evaluation.total_true_positives) + np.sum(self.evaluation.total_false_positives) == 0:
            self.evaluation.micro_precision = 1.0
        else:
            self.evaluation.micro_precision = self.evaluation.total_true_positives / (
                self.evaluation.total_true_positives + self.evaluation.total_false_positives)

        self.evaluation.micro_recall = self.evaluation.total_true_positives / (
            self.evaluation.total_true_positives + self.evaluation.total_false_negatives)

        if self.evaluation.micro_precision + self.evaluation.micro_recall > 0:
            self.evaluation.micro_f1 = 2 * (self.evaluation.micro_precision * self.evaluation.micro_recall) / (
                self.evaluation.micro_precision + self.evaluation.micro_recall)
        else:
            self.evaluation.micro_f1 = 0.0

        self.evaluation.macro_precision /= self.evaluation.n_samples
        self.evaluation.macro_recall /= self.evaluation.n_samples

        if self.evaluation.macro_precision + self.evaluation.macro_recall > 0:
            self.evaluation.macro_f1 = 2 * (self.evaluation.macro_precision * self.evaluation.macro_recall) / (
                self.evaluation.macro_precision + self.evaluation.macro_recall)
        else:
            self.evaluation.macro_f1 = 0.0


        return self.evaluation