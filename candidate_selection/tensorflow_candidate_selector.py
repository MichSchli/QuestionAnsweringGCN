import tensorflow as tf
import numpy as np

from helpers.static import Static


class TensorflowCandidateSelector:

    candidate_neighborhood_generator = None
    gold_generator = None
    model = None
    facts = None

    # To be moved to configuration file
    epochs = 20
    batch_size = 30

    def __init__(self, model, candidate_neighborhood_generator):
        self.candidate_neighborhood_generator = candidate_neighborhood_generator
        self.model = model

    def update_setting(self, setting_string, value):
        if setting_string == "epochs":
            self.epochs = int(value)
        elif setting_string == "batch_size":
            self.batch_size = int(value)
        elif setting_string == "facts":
            self.facts = value

        self.model.update_setting(setting_string, value)

    def initialize(self):
        tf.reset_default_graph()
        self.model.initialize()
        self.model.prepare_tensorflow_variables(mode='train')

        model_loss = self.model.get_loss_graph()
        parameters_to_optimize = tf.trainable_variables()
        opt_func = tf.train.AdamOptimizer(learning_rate=0.01)
        grad_func = tf.gradients(model_loss, parameters_to_optimize)
        self.optimize_func = opt_func.apply_gradients(zip(grad_func, parameters_to_optimize))
        init_op = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_op)

    def valid_example(self, elements):
        return self.model.validate_example(elements)

    def iterate_in_batches(self, iterator, validate_batches=False, batch_size = None):
        batch_dict = {}

        if batch_size is None:
            batch_size = self.batch_size

        index = 0
        for example in iterator:
            if validate_batches and not self.model.validate_example(example):
                continue

            for k,v in example.items():
                if k not in batch_dict:
                    batch_dict[k] = [None]*batch_size
                batch_dict[k][index] = v

            index += 1
            if index == batch_size:
                yield batch_dict
                index = 0
                batch_dict = {}

        if index != 0:
            yield {k:v[:index] for k,v in batch_dict.items()}

    def train(self, train_file_iterator, epochs=None):
        if epochs is None:
            epochs = self.epochs

        model_loss = self.model.get_loss_graph()

        for epoch in range(epochs):
            Static.logger.write("Starting epoch " + str(epoch), verbosity_priority=4)
            epoch_iterator = train_file_iterator.iterate()
            epoch_iterator = self.candidate_neighborhood_generator.enrich(epoch_iterator)

            batch_iterator = self.iterate_in_batches(epoch_iterator, validate_batches=True)
            for i,batch in enumerate(batch_iterator):
                self.model.get_preprocessor().process(batch)

                assignment_dict = self.model.handle_variable_assignment(batch, mode='train')
                result = self.sess.run([self.optimize_func, model_loss], feed_dict=assignment_dict)
                loss = result[1]

                Static.logger.write("Loss at batch "+str(i) + ": " + str(loss), verbosity_priority=3)

    def predict(self, test_file_iterator):
        example_iterator = test_file_iterator.iterate()
        example_iterator = self.candidate_neighborhood_generator.enrich(example_iterator)
        #batch_iterator = self.iterate_in_batches(epoch_iterator, batch_size=1)

        model_prediction = self.model.get_prediction_graph()
        model_loss = self.model.get_loss_graph()

        for example in example_iterator:
            can_be_predicted = self.model.validate_example(example)
            if not can_be_predicted:
                yield []
                continue

            as_batch = {k:[v] for k,v in example.items()}
            self.model.get_preprocessor().process(as_batch)

            assignment_dict = self.model.handle_variable_assignment(as_batch, mode='predict')
            predictions, loss = self.sess.run([model_prediction, model_loss], feed_dict=assignment_dict)

            for i, prediction in enumerate(predictions):
                best_predictions = np.where(prediction[0] > .5)[0]
                output = []
                for prediction in best_predictions:
                    output.extend(self.model.retrieve_entities(i,prediction))

                yield output
