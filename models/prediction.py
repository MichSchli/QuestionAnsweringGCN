import numpy as np


class Prediction():

    label_list = None
    score_list = None

    def __init__(self):
        self.label_list = []
        self.score_list = []

    def add_predictions(self, labels, scores):
        self.label_list.extend(labels)
        self.score_list.extend(scores)

    def get_predictions(self, cutoff):
        predictions = []
        for label,score in zip(self.label_list, self.score_list):
            if score >= cutoff:
                predictions.append(label)

        return np.unique(predictions)