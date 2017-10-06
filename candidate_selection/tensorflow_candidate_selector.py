import tensorflow as tf
import numpy as np


class TensorflowCandidateSelector:

    candidate_neighborhood_generator = None
    gold_generator = None
    model = None
    facts = None

    # To be moved to configuration file
    epochs = 20
    batch_size = 30

    def __init__(self, model, candidate_neighborhood_generator, gold_generator, facts):
        self.candidate_neighborhood_generator = candidate_neighborhood_generator
        self.gold_generator = gold_generator
        self.model = model
        self.facts = facts

    def valid_example(self, elements):
        return self.model.validate_example(elements)

    def iterate_in_batches(self, iterators, validate_batches=True):
        batch = [[None]*self.batch_size for _ in iterators]

        index = 0
        for elements in zip(*iterators):
            if validate_batches and not self.valid_example(elements):
                print("invalidated "+str(index))
                continue

            for j, element in enumerate(elements):
                batch[j][index] = element
            index += 1

            if index == self.batch_size:
                yield batch
                index = 0

        if index != 0:
            yield [b[:index] for b in batch]

    def train(self, training_file):
        self.model.prepare_tensorflow_variables(mode='train')
        model_loss = self.model.get_loss_graph()
        parameters_to_optimize = tf.trainable_variables()
        opt_func = tf.train.AdamOptimizer(learning_rate=0.01)
        grad_func = tf.gradients(model_loss, parameters_to_optimize)
        optimize_func = opt_func.apply_gradients(zip(grad_func, parameters_to_optimize))
        init_op = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_op)

        for epoch in range(self.epochs):
            print("Starting epoch: "+str(epoch))

            candidate_iterator = self.candidate_neighborhood_generator.parse_file(training_file)
            aux_iterators = self.model.get_aux_iterators()
            label_iterator = self.gold_generator.parse_file(training_file)

            all_iterators = [candidate_iterator] + aux_iterators + [label_iterator]

            batch_iterator = self.iterate_in_batches(all_iterators)

            for batch in batch_iterator:
                preprocessed_batch = self.model.preprocess(batch, mode='train')
                assignment_dict = self.model.handle_variable_assignment(preprocessed_batch, mode='train')
                result = self.sess.run([optimize_func, model_loss], feed_dict=assignment_dict)
                loss = result[1]
                print(loss)

    def parse_file(self, filename):
        self.batch_size = 1
        candidate_iterator = self.candidate_neighborhood_generator.parse_file(filename)
        aux_iterators = self.model.get_aux_iterators()
        label_iterator = self.gold_generator.parse_file(filename)
        all_iterators = [candidate_iterator] + aux_iterators + [label_iterator]
        batch_iterator = self.iterate_in_batches(all_iterators, validate_batches=False)

        #self.model.prepare_variables(mode='predict')

        model_prediction = self.model.get_prediction_graph()
        model_loss = self.model.get_loss_graph()

        for batch in batch_iterator:
            print(" ".join([w[1] for w in batch[1][0][0]]))
            preprocessed_batch = self.model.preprocess(batch, mode='predict')
            assignment_dict = self.model.handle_variable_assignment(preprocessed_batch, mode='predict')
            predictions, loss = self.sess.run([model_prediction, model_loss], feed_dict=assignment_dict)

            #print("Loss was: " + str(loss))

            for i, prediction in enumerate(predictions):
                best_predictions = np.where(prediction[0] > .3)[0]
                output = []
                for prediction in best_predictions:
                    output.extend(self.model.retrieve_entities(i,prediction))

                yield output
