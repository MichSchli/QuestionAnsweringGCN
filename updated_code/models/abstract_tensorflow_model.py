from models.prediction import Prediction
import tensorflow as tf


class AbstractTensorflowModel:


    components = None
    graphs = None

    def __init__(self):
        self.graphs = {}
        self.components = []

    def initialize(self):
        init_op = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_op)

    def add_component(self, component):
        self.components.append(component)

    def handle_variable_assignment(self, batch, mode):
        for component in self.components:
            component.handle_variable_assignment(batch, mode)

    def predict(self, batch):
        model_prediction = self.get_prediction_graph()
        self.handle_variable_assignment(batch, "test")
        predictions = self.sess.run(model_prediction, feed_dict=self.get_assignment_dict())

        example_begin_index = 0
        for example in batch.examples:
            scores = predictions[example_begin_index:example_begin_index + example.count_vertices()]
            example_begin_index += example.count_vertices()

            prediction = Prediction()
            vertex_indexes = example.graph.get_entity_vertices()
            vertex_labels = [example.graph.map_to_label(v) for v in vertex_indexes]

            prediction.add_predictions(vertex_labels, scores)
            example.prediction = prediction

    def get_assignment_dict(self):
        assignment_dict = {}
        for component in self.components:
            for k, v in component.variables.items():
                assignment_dict[v] = component.variable_assignments[k]
        return assignment_dict

    def get_loss_graph(self, mode="train"):
        if mode not in self.graphs:
            self.graphs[mode] = self.compute_entity_scores(mode)

        return self.loss.compute(self.graphs[mode])

    def get_prediction_graph(self, mode="predict"):
        return tf.nn.sigmoid(self.compute_entity_scores(mode))