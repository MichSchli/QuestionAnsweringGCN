import numpy as np

from example_reader.graph_reader.database_interface.data_interface.FreebaseInterface import FreebaseInterface
from model_tester.evaluator.evaluation import Evaluation


class RelationPredictionFreebaseEntityEvaluator:

    evaluation = None

    def __init__(self, relation_index):
        self.relation_index = relation_index
        self.freebase_interface = FreebaseInterface()

    def begin_evaluation(self, steps=[0.5]):
        self.evaluations = [(step, Evaluation(default_method="macro")) for step in steps]

    def add_prediction(self, example):
        for step, evaluation in self.evaluations:
            self.add_predictions_to_step(example, step, evaluation)

    def add_predictions_to_step(self, example, step, evaluation):
        true_labels = example.get_gold_labels()

        predicted_labels = example.prediction
        pred_target = np.argmax(predicted_labels)
        pred_edge = self.relation_index.from_index(pred_target)

        if pred_edge == "<unknown>":
            evaluation.macro_precision += 1.0
            evaluation.macro_recall += 0.0
            evaluation.macro_f1 += 0.0
            evaluation.n_samples += 1
            return

        max_precision = 0
        max_recall = 0
        max_f1 = 0

        for c in example.mentions:
            centroid = c.entity_label
            edge_parts = pred_edge.split("|")

            first_edge = edge_parts[0].strip()

            forward = True
            singular = first_edge.endswith(".1") or first_edge.endswith(".2")

            if first_edge.endswith(".1"):
                first_edge = first_edge[:-2]
            elif first_edge.endswith(".2"):
                first_edge = first_edge[:-2]
                forward = False
            elif first_edge.endswith(".inverse"):
                first_edge = first_edge[:-8]
            else:
                forward = False

            retrieved = self.freebase_interface.get_entities([centroid], first_edge, forward)

            if singular:
                names = [self.freebase_interface.get_name(r) for r in retrieved]
                full_predictions = [n if len(n) > 0 else [r] for n,r in zip(names, retrieved)]
                if len(full_predictions) > 0:
                    full_predictions = np.concatenate(full_predictions)
            else:
                second_edge = edge_parts[1].strip()
                forward = True

                if second_edge.endswith(".inverse"):
                    second_edge = second_edge[:-8]
                    forward = False

                second_retrieved = self.freebase_interface.get_entities(retrieved, second_edge, forward)

                names = [self.freebase_interface.get_name(r) for r in second_retrieved]
                full_predictions = [n if len(n) > 0 else [r] for n,r in zip(names, second_retrieved)]
                if len(full_predictions) > 0:
                    full_predictions = np.concatenate(full_predictions)

            print(full_predictions)
            print(true_labels)

            true_positives = np.isin(full_predictions, true_labels)
            false_positives = np.logical_not(true_positives)
            false_negatives = np.isin(true_labels, full_predictions, invert=True)

            if np.sum(true_positives) + np.sum(false_positives) == 0:
                inner_precision = 1.0
            else:
                inner_precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))

            if np.sum(true_positives) + np.sum(false_negatives) == 0:
                inner_recall = 1.0
            else:
                inner_recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))

            if inner_precision + inner_recall > 0:
                inner_f1 = 2 * (inner_precision * inner_recall) / (inner_precision + inner_recall)
            else:
                inner_f1 = 0

            print(inner_precision)
            print(inner_recall)
            print(inner_f1)

            if inner_f1 > max_f1:
                max_precision = inner_precision
                max_recall = inner_recall
                max_f1 = inner_f1

        evaluation.macro_precision += max_precision
        evaluation.macro_recall += max_recall
        evaluation.macro_f1 += max_f1
        evaluation.n_samples += 1

    def compute_final_scores(self, evaluation):
        pass

    def final_scores(self):
        for step, evaluation in self.evaluations:
            self.compute_final_scores(evaluation)

        return self.evaluations