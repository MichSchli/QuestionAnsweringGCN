from models.prediction import Prediction
import tensorflow as tf


class AbstractTensorflowModel:


    components = None
    graphs = None
    optimize_func = None

    def __init__(self):
        self.graphs = {}
        self.components = []

    def initialize(self):
        self.compute_update_graph()
        init_op = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_op)

    def add_component(self, component):
        self.components.append(component)

    def handle_variable_assignment(self, batch, mode):
        for component in self.components:
            component.handle_variable_assignment(batch, mode)

    def compute_update_graph(self):
        if self.optimize_func is None:
            parameters_to_optimize = tf.trainable_variables()
            opt_func = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients = tf.gradients(self.get_loss_graph(), parameters_to_optimize)
            grad_func = tf.clip_by_global_norm(gradients, self.gradient_clipping)[0]
            self.optimize_func = opt_func.apply_gradients(zip(grad_func, parameters_to_optimize))
        return self.optimize_func

    def update(self, batch):
        model_loss = self.get_loss_graph()
        model_update = self.compute_update_graph()

        self.handle_variable_assignment(batch, "train")
        loss, _ = self.sess.run([model_loss, model_update], feed_dict=self.get_assignment_dict())

        return loss

    def predict_batch(self, batch):
        model_prediction = self.get_prediction_graph()
        self.handle_variable_assignment(batch, "test")
        predictions = self.sess.run(model_prediction, feed_dict=self.get_assignment_dict())
        return predictions

    def get_assignment_dict(self):
        assignment_dict = {}
        for component in self.components:
            for k, v in component.variables.items():
                assignment_dict[v] = component.variable_assignments[k]
        return assignment_dict

    loss_function = None

    def get_loss_graph(self, mode="train"):
        if mode not in self.graphs:
            self.graphs[mode] = self.compute_entity_scores(mode)

        if self.loss_function is None:
            self.loss_function = self.loss.compute_loss(self.graphs[mode])
            self.loss_function += sum(component.get_regularization() for component in self.components)

        return self.loss_function

    predict_function = None

    def get_prediction_graph(self, mode="predict"):
        if mode not in self.graphs:
            self.graphs[mode] = self.compute_entity_scores(mode)

        if self.predict_function is None:
            self.predict_function = self.loss.compute_prediction(self.graphs[mode])

        return self.predict_function