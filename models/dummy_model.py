from models.prediction import Prediction


class DummyModel:

    def __init__(self):
        pass

    def predict(self, potential_batch):
        for example in potential_batch.get_examples():
            prediction = Prediction()

            vertex_indexes = example.graph.get_entity_vertices()
            vertex_labels = [example.graph.map_to_label(v) for v in vertex_indexes]
            scores = [1.0 for _ in vertex_labels]

            prediction.add_predictions(vertex_labels, scores)

            example.prediction = prediction