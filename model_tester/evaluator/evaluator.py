import numpy as np

from model_tester.evaluator.evaluation import Evaluation


class Evaluator:

    evaluation = None

    def __init__(self, relation_index):
        self.relation_index = relation_index

    def begin_evaluation(self, steps=[0.5]):
        self.evaluations = [(step, Evaluation(default_method="macro")) for step in steps]

    def add_prediction(self, example):
        for step, evaluation in self.evaluations:
            self.add_predictions_to_step(example, step, evaluation)

    def add_predictions_to_step(self, example, step, evaluation):
        true_labels = example.get_gold_labels()
        predicted_labels = example.get_predicted_labels(step)

        if true_labels.shape[0] == 0:
            print("ERROR. ERROR.")
            exit()

        print(example)
        indexes = sorted(range(len(example.prediction.score_list)), key=lambda x: example.prediction.score_list[x], reverse=True)
        for pred_index in indexes[:min(len(predicted_labels), 5)]:
            print(example.prediction.label_list[pred_index]+ " | score=" + str(example.prediction.score_list[pred_index]))
            print(example.graph.entity_centroid_paths[pred_index])

        true_positives = np.isin(predicted_labels, true_labels)
        false_positives = np.logical_not(true_positives)
        false_negatives = np.isin(true_labels, predicted_labels, invert=True)

        evaluation.total_true_positives += np.sum(true_positives)
        evaluation.total_false_positives += np.sum(false_positives)
        evaluation.total_false_negatives += np.sum(false_negatives)

        if np.sum(true_positives) + np.sum(false_positives) == 0:
            inner_precision = 1.0
        else:
            inner_precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))

        if np.sum(true_positives) + np.sum(false_negatives) == 0:
            inner_recall = 1.0
        else:
            inner_recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))

        evaluation.macro_precision += inner_precision
        evaluation.macro_recall += inner_recall

        if inner_precision + inner_recall > 0:
            inner_f1 = 2 * (inner_precision * inner_recall) / (inner_precision + inner_recall)
        else:
            inner_f1 = 0

        evaluation.macro_f1 += inner_f1
        evaluation.n_samples += 1

        #print(evaluation.temporary_summary())

    def compute_final_scores(self, evaluation):
        if np.sum(evaluation.total_true_positives) + np.sum(evaluation.total_false_positives) == 0:
            evaluation.micro_precision = 1.0
        else:
            evaluation.micro_precision = evaluation.total_true_positives / (
                evaluation.total_true_positives + evaluation.total_false_positives)

        evaluation.micro_recall = evaluation.total_true_positives / (
            evaluation.total_true_positives + evaluation.total_false_negatives)

        if evaluation.micro_precision + evaluation.micro_recall > 0:
            evaluation.micro_f1 = 2 * (evaluation.micro_precision * evaluation.micro_recall) / (
                evaluation.micro_precision + evaluation.micro_recall)
        else:
            evaluation.micro_f1 = 0.0

        evaluation.macro_precision /= evaluation.n_samples
        evaluation.macro_recall /= evaluation.n_samples
        evaluation.macro_f1 /= evaluation.n_samples

    def final_scores(self):
        for step, evaluation in self.evaluations:
            self.compute_final_scores(evaluation)

        return self.evaluations