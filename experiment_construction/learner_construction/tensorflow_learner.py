import tensorflow as tf
import numpy as np
from helpers.static import Static
import time


class TensorflowModel:

    epochs = None
    batch_size = None
    project_names = False

    preprocessor = None
    candidate_generator = None
    model = None

    learning_rate = None

    def update_setting(self, setting_string, value):
        if setting_string == "epochs":
            self.epochs = int(value)
        elif setting_string == "batch_size":
            self.batch_size = int(value)
        elif setting_string == "project_name":
            self.project_names = True if value == "True" else False
        elif setting_string == "learning_rate":
            self.learning_rate = float(value)

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def set_candidate_generator(self, candidate_generator):
        self.candidate_generator = candidate_generator

    def set_candidate_selector(self, candidate_selector):
        self.model = candidate_selector

    def initialize(self):
        tf.reset_default_graph()
        self.model.initialize()
        self.model.prepare_tensorflow_variables(mode='train')

        self.model_loss = self.model.get_loss_graph()
        parameters_to_optimize = tf.trainable_variables()
        opt_func = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grad_func = tf.gradients(self.model_loss, parameters_to_optimize)
        self.optimize_func = opt_func.apply_gradients(zip(grad_func, parameters_to_optimize))
        init_op = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_op)

    def iterate_in_batches(self, iterator, validate_batches=False, batch_size = None):
        batch_dict = {}

        if batch_size is None:
            batch_size = self.batch_size

        index = 0
        for example in iterator:
            print(index)
            if validate_batches and not self.model.validate_example(example):
                print("invalid")
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

    def project_gold(self, iterator):
        for example in iterator:
            candidates = example["neighborhood"].get_vertices(type="entities")
            target_vertices = example["gold_entities"]
            projected_target_vertices = np.array([example["neighborhood"].to_index(e) for e in target_vertices if example["neighborhood"].has_index(e)])
            #target_vertices_in_candidates = np.isin(projected_target_vertices, candidates)

            #print(projected_target_vertices.shape[0])
            #print(projected_target_vertices)
            if projected_target_vertices.shape[0] > 0:
                example["gold_entities"] = projected_target_vertices
                yield example

    def project_gold_to_index(self, iterator):
        for example in iterator:
            names = example["gold_entities"]
            graph = example["neighborhood"]
            gold_list = []
            for name in names:
                if graph.has_index(name):
                    gold_list.append(graph.to_index(name))

            # TODO CHECK SOMEWHERE ELSE
            if len(gold_list) == 0:
                continue

            gold_list = np.array(gold_list).astype(np.int32)
            example["gold_entities"] = gold_list
            yield example

    def project_from_name_wrapper(self, iterator, skip=True):
        for example in iterator:
            names = example["gold_entities"]
            graph = example["neighborhood"]
            name_projection_dictionary = graph.get_inverse_name_connections(names)

            gold_list = []
            for name,l in name_projection_dictionary.items():
                if len(l) > 0:
                    gold_list.extend(l)
                elif graph.has_index(name):
                    gold_list.append(graph.to_index(name))

            # TODO CHECK SOMEWHERE ELSE
            if len(gold_list) == 0:
                print("name " + str(names) + " does not match anything, discarding")
                if not skip:
                    yield example
                    
                continue

            gold_list = np.array(gold_list).astype(np.int32)
            #print(example["neighborhood"].entity_vertices.shape[0])
            print("projected " + str(example["gold_entities"]) + " to " + str(gold_list))
            example["gold_entities"] = gold_list
            yield example

    def train(self, train_file_iterator, start_epoch=0, epochs=None):
        if epochs is None:
            epochs = self.epochs

        model_prediction = self.model.get_prediction_graph()

        for epoch in range(start_epoch, start_epoch+epochs):
            Static.logger.write("Starting epoch " + str(epoch), "training", "iteration_messages")
            epoch_iterator = train_file_iterator.iterate()
            epoch_iterator = self.candidate_generator.enrich(epoch_iterator)
            #epoch_iterator = self.project_gold(epoch_iterator)

            if self.project_names:
                epoch_iterator = self.project_from_name_wrapper(epoch_iterator)
            else:
                epoch_iterator = self.project_gold_to_index(epoch_iterator)

            batch_iterator = self.iterate_in_batches(epoch_iterator, validate_batches=False)
            for i,batch in enumerate(batch_iterator):
                #print("asdf")
                self.preprocessor.process(batch, mode="train")
                assignment_dict = self.model.handle_variable_assignment(batch, mode='train')
                result = self.sess.run([self.optimize_func, self.model_loss, model_prediction], feed_dict=assignment_dict)
                loss = result[1]
                best_predictions = np.where(result[2] > .5)[2]
                print(best_predictions)

                for prediction in best_predictions:
                    if Static.logger.should_log("testing", "paths"):
                        Static.logger.write(batch_iterator["neighborhood"][0].get_paths_to_neighboring_centroid(prediction), "testing", "paths")
                    if batch_iterator["neighborhood"].has_name(prediction):
                        print(batch_iterator["neighborhood"][0].get_name(prediction))
                    else:
                        print(batch_iterator["neighborhood"][0].from_index(prediction))

                time.sleep(5)

                Static.logger.write("Loss at batch "+str(i) + ": " + str(loss), "training", "iteration_loss")


    def predict(self, test_file_iterator):
        example_iterator = test_file_iterator.iterate()
        example_iterator = self.candidate_generator.enrich(example_iterator)

        if self.project_names:
            example_iterator = self.project_from_name_wrapper(example_iterator, skip=False)
        else:
            example_iterator = self.project_gold_to_index(example_iterator)

        model_prediction = self.model.get_prediction_graph()
        for j,example in enumerate(example_iterator):

            if example["neighborhood"].get_vertices(type="entities").shape[0] == 0:
                yield []
                continue

            print("example "+str(j)+":\n - - - - - - - ")

            as_batch = {k:[v] for k,v in example.items()}
            self.preprocessor.process(as_batch, mode='predict')

            assignment_dict = self.model.handle_variable_assignment(as_batch, mode='predict')
            predictions = self.sess.run(model_prediction, feed_dict=assignment_dict)

            for i, prediction in enumerate(predictions):
                #print(prediction)
                best_predictions = np.where(prediction[0] > .5)[0]
                print(best_predictions)
                output = []

                for prediction in best_predictions:
                    if Static.logger.should_log("testing", "paths"):
                        Static.logger.write(example["neighborhood"].get_paths_to_neighboring_centroid(prediction), "testing", "paths")
                    if example["neighborhood"].has_name(prediction):
                        output.append(example["neighborhood"].get_name(prediction))
                    else:
                        output.append(example["neighborhood"].from_index(prediction))

                #print([example["neighborhood"].from_index(i) for i in example["gold_entities"]])
                print(output)
                print("=====")

                yield output
