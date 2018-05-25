import numpy as np

from model_tester.evaluator.evaluation import Evaluation


class RelationPredictionEvaluator:

    evaluation = None

    def begin_evaluation(self, steps=[0.5]):
        self.evaluations = [(step, Evaluation(default_method="micro")) for step in steps]

    def add_prediction(self, example):
        for step, evaluation in self.evaluations:
            self.add_predictions_to_step(example, step, evaluation)

    def add_predictions_to_step(self, example, step, evaluation):
        true_labels = example.get_gold_relation_vector()
        predicted_labels = example.prediction

        target = np.argmax(true_labels)
        pred_target = np.argmax(predicted_labels)

        if target == pred_target:
            evaluation.total_true_positives += 1
        else:
            evaluation.total_false_positives += 1
            evaluation.total_false_negatives += 1

        evaluation.n_samples += 1

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

    def final_scores(self):
        for step, evaluation in self.evaluations:
            self.compute_final_scores(evaluation)

        return self.evaluations