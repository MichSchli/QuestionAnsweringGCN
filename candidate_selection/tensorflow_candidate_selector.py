import tensorflow as tf
import numpy as np


class TensorflowCandidateSelector:

    candidate_neighborhood_generator = None
    gold_generator = None
    model = None
    facts = None

    # To be moved to configuration file
    epochs = 1000
    batch_size = 2

    def __init__(self, model, candidate_neighborhood_generator, gold_generator, facts):
        self.candidate_neighborhood_generator = candidate_neighborhood_generator
        self.gold_generator = gold_generator
        self.model = model
        self.facts = facts

    def iterate_in_batches(self, iterator):
        batch = [None]*self.batch_size

        index = 0
        for element in iterator:
            batch[index] = element
            index += 1

            if index == self.batch_size:
                yield batch
                index = 0

        if index != 0:
            yield batch[:index]

    def train(self, training_file):
        self.model.prepare_variables(mode='train')
        model_loss = self.model.get_loss_graph()
        parameters_to_optimize = self.model.get_optimizable_parameters()
        opt_func = tf.train.AdamOptimizer(learning_rate=0.001)
        grad_func = tf.gradients(model_loss, parameters_to_optimize)
        optimize_func = opt_func.apply_gradients(zip(grad_func, parameters_to_optimize))
        init_op = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_op)

        for epoch in range(self.epochs):
            print("Starting epoch: "+str(epoch))

            candidate_iterator = self.candidate_neighborhood_generator.parse_file(training_file)
            batch_candidate_iterator = self.iterate_in_batches(candidate_iterator)

            aux_iterators = self.model.get_aux_iterators()
            aux_batch_iterators = [self.iterate_in_batches(i) for i in aux_iterators]

            label_iterator = self.gold_generator.parse_file(training_file)
            batch_label_iterator = self.iterate_in_batches(label_iterator)

            all_iterators = [batch_candidate_iterator] + aux_batch_iterators + [batch_label_iterator]

            for batch in zip(*all_iterators):
                preprocessed_batch = self.model.preprocess(batch, mode='train')
                assignment_dict = self.model.handle_variable_assignment(preprocessed_batch, mode='train')
                result = self.sess.run([optimize_func, model_loss], feed_dict=assignment_dict)
                loss = result[1]
                print(loss)

    def predict(self, filename):
        candidate_iterator = self.candidate_neighborhood_generator.parse_file(filename)
        batch_iterator = self.iterate_in_batches(candidate_iterator)

        aux_iterators = self.model.get_aux_iterators()
        aux_batch_iterators = [self.iterate_in_batches(i) for i in aux_iterators]
        all_iterators = [batch_iterator] + aux_batch_iterators

        #self.model.prepare_variables(mode='predict')

        model_prediction = self.model.get_prediction_graph()

        for batch in zip(*all_iterators):
            preprocessed_batch = self.model.preprocess(batch, mode='predict')
            assignment_dict = self.model.handle_variable_assignment(preprocessed_batch, mode='predict')
            predictions = self.sess.run(model_prediction, feed_dict=assignment_dict)

            for prediction in predictions:
                best_prediction = np.argmax(prediction)
                output = self.model.retrieve_entities(best_prediction)
                yield output
