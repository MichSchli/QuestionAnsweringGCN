class RelationPredictionOracle:

    def __init__(self):
        pass

    def initialize(self):
        pass

    def update(self, batch):
        return 0

    def predict_batch(self, batch):
        predictions = batch.get_relation_class_labels()

        return predictions