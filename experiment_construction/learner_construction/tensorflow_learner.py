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
    gradient_clipping = None

    average_loss_over_n = None

    def update_setting(self, setting_string, value):
        if setting_string == "epochs":
            self.epochs = int(value)
        elif setting_string == "batch_size":
            self.batch_size = int(value)
        elif setting_string == "gradient_clipping":
            self.gradient_clipping = float(value)
        elif setting_string == "project_name":
            self.project_names = True if value == "True" else False
        elif setting_string == "learning_rate":
            self.learning_rate = float(value)
        elif setting_string == "average_loss_over_n":
            self.average_loss_over_n = int(value)

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
        gradients = tf.gradients(self.model_loss, parameters_to_optimize)
        grad_func = tf.clip_by_global_norm(gradients, self.gradient_clipping)[0]
        #grad_func = [tf.clip_by_norm(grad, self.gradient_clipping) if grad is not None else grad for grad in gradients]
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

    def project_gold_to_index(self, iterator):
        for example in iterator:
            names = example["gold_entities"]
            graph = example["neighborhood"]
            gold_list = []
            for name in names:
                if graph.has_index(name):
                    gold_list.extend(graph.to_index(name))

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
                    gold_list.extend(graph.to_index(name))

            # TODO CHECK SOMEWHERE ELSE
            if len(gold_list) == 0:
                #print("name " + str(names) + " does not match anything, discarding")
                if not skip:
                    yield example
                    
                continue

            gold_list = np.array(gold_list).astype(np.int32)
            #print(example["neighborhood"].entity_vertices.shape[0])
            #print("projected " + str(example["gold_entities"]) + " to " + str(gold_list))
            example["gold_entities"] = gold_list
            yield example

    def split_graphs(self, iterator):
        for example in iterator:
            graph = example["neighborhood"]
            example["neighborhood"] = graph.get_split_graph()
            scores = example["sentence_entity_map"][:,3].astype(np.float32)
            example["neighborhood"].propagate_scores(scores)

            yield example

    def train(self, train_file_iterator, start_epoch=0, epochs=None):
        if epochs is None:
            epochs = self.epochs

        model_prediction = self.model.get_prediction_graph()

        for epoch in range(start_epoch, start_epoch+epochs):
            Static.logger.write("Starting epoch " + str(epoch), "training", "iteration_messages")
            epoch_iterator = train_file_iterator.iterate(shuffle=True)
            epoch_iterator = self.candidate_generator.enrich(epoch_iterator)
            epoch_iterator = self.split_graphs(epoch_iterator)
            #epoch_iterator = self.project_gold(epoch_iterator)

            if self.project_names:
                epoch_iterator = self.project_from_name_wrapper(epoch_iterator)
            else:
                epoch_iterator = self.project_gold_to_index(epoch_iterator)

            batch_iterator = self.iterate_in_batches(epoch_iterator, validate_batches=False)
            loss_counter = 0
            for i,batch in enumerate(batch_iterator):
                #print("asdf")
                self.preprocessor.process(batch, mode="train")
                assignment_dict = self.model.handle_variable_assignment(batch, mode='train')
                result = self.sess.run([self.optimize_func, self.model_loss, model_prediction], feed_dict=assignment_dict)
                loss = result[1]
                #best_predictions = np.where(result[2] > .5)[2]
                #print("\n========================")
                #print(batch["gold_entities"][0])
                #print(best_predictions)

                #for prediction in sorted(best_predictions, key=lambda x: result[2][0][0][x], reverse=True)[:10]:
                #    print(result[2][0][0][prediction])
                #    print(batch["neighborhood"][0].get_paths_to_neighboring_centroid(prediction), "testing", "paths")
                #    if batch["neighborhood"][0].has_name(prediction):
                #        print(batch["neighborhood"][0].get_name(prediction))
                #    else:
                #        print(batch["neighborhood"][0].from_index(prediction))

                #time.sleep(2)
                loss_counter += loss

                if (i+1) % self.average_loss_over_n == 0:
                    Static.logger.write("Average Loss for batch "+str(i+1-self.average_loss_over_n) + " to " + str(i) + ": " + str(loss_counter/self.average_loss_over_n), "training", "iteration_loss")
                    loss_counter = 0
                #time.sleep(2)

    def predict(self, test_file_iterator):
        example_iterator = test_file_iterator.iterate()
        example_iterator = self.candidate_generator.enrich(example_iterator)
        example_iterator = self.split_graphs(example_iterator)

        if self.project_names:
            example_iterator = self.project_from_name_wrapper(example_iterator, skip=False)
        else:
            example_iterator = self.project_gold_to_index(example_iterator)

        model_prediction = self.model.get_prediction_graph()
        batch_iterator = self.iterate_in_batches(example_iterator, validate_batches=False)
        for j,batch in enumerate(batch_iterator):
            print("batch "+str(j)+":\n - - - - - - - ")

            self.preprocessor.process(batch, mode='predict')

            assignment_dict = self.model.handle_variable_assignment(batch, mode='predict')
            predictions = self.sess.run(model_prediction, feed_dict=assignment_dict)

            for i, prediction in enumerate(predictions):
                l = list(sorted(enumerate(prediction[0]), key=lambda x: x[1], reverse=True))

                for index,prob in l[:5]:
                    if prob > 0.5:
                        print(batch["neighborhood"][i].get_paths_to_neighboring_centroid(index))

                yield [(batch["neighborhood"][i].from_index_with_names(index),prob) for index,prob in l]
                continue
