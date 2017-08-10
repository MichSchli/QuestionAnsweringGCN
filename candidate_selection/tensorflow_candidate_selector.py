import tensorflow as tf
import numpy as np


class TensorflowCandidateSelector:

    candidate_neighborhood_generator = None
    gold_generator = None
    model = None

    # To be moved to configuration file
    batch_size = 5

    def __init__(self, model, candidate_neighborhood_generator, gold_generator):
        self.candidate_neighborhood_generator = candidate_neighborhood_generator
        self.gold_generator = gold_generator
        self.model = model

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


    def parse_file(self, filename):
        candidate_iterator = self.candidate_neighborhood_generator.parse_file(filename)
        batch_iterator = self.iterate_in_batches(candidate_iterator)

        model_input_variables = self.model.get_prediction_input()
        model_prediction = self.model.get_prediction_graph()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            for candidate_graph_batch in batch_iterator:
                preprocessed_batch = self.model.preprocess(candidate_graph_batch)
                #predictions = self.model.predict(preprocessed_batch)
                predictions = sess.run(model_prediction, feed_dict={model_input_variables[0]: preprocessed_batch[0],
                                                                    model_input_variables[1]: preprocessed_batch[1]})


                for prediction in predictions:
                    best_prediction = np.argmax(prediction)
                    output = self.model.retrieve_entities(best_prediction)
                    yield output
