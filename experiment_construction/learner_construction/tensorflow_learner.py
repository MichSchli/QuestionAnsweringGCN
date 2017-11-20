import tensorflow as tf
import numpy as np
from helpers.static import Static


class TensorflowModel:

    epochs = None
    batch_size = None
    project_names = False

    preprocessor = None
    candidate_generator = None
    model = None

    def update_setting(self, setting_string, value):
        if setting_string == "epochs":
            self.epochs = int(value)
        elif setting_string == "batch_size":
            self.batch_size = int(value)
        elif setting_string == "project_name":
            self.project_names = True if value == "True" else False

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

        model_loss = self.model.get_loss_graph()
        parameters_to_optimize = tf.trainable_variables()
        opt_func = tf.train.AdamOptimizer(learning_rate=0.01)
        grad_func = tf.gradients(model_loss, parameters_to_optimize)
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

    def project_gold(self, iterator):
        for example in iterator:
            candidates = example["neighborhood"].get_vertices(type="entities")
            target_vertices = example["gold_entities"]
            projected_target_vertices = np.array([example["neighborhood"].to_index(e) for e in target_vertices if example["neighborhood"].has_index(e)])
            #target_vertices_in_candidates = np.isin(projected_target_vertices, candidates)

            print(projected_target_vertices.shape[0])
            print(projected_target_vertices)
            if projected_target_vertices.shape[0] > 0:
                example["gold_entities"] = projected_target_vertices
                yield example

    def project_from_name_wrapper(self, iterator):
        for example in iterator:
            names = example["gold_entities"]
            graph = example["neighborhood"]
            name_projection_dictionary = graph.get_inverse_name_connections(names)

            gold_list = []
            for name,l in name_projection_dictionary.items():
                if len(l) > 0:
                    gold_list.extend(l)
                else:
                    gold_list.append(name)

            # TODO CHECK SOMEWHERE ELSE
            if len(gold_list) == 0:
                continue

            gold_list = np.array(gold_list).astype(np.int32)
            print(example["neighborhood"].entity_vertices.shape[0])
            print("Swapping " + str(example["gold_entities"]) + " for " + str(gold_list))

            example["gold_entities"] = gold_list
            yield example

    def train(self, train_file_iterator, epochs=None):
        if epochs is None:
            epochs = self.epochs

        model_loss = self.model.get_loss_graph()

        for epoch in range(epochs):
            Static.logger.write("Starting epoch " + str(epoch), verbosity_priority=4)
            epoch_iterator = train_file_iterator.iterate()
            epoch_iterator = self.candidate_generator.enrich(epoch_iterator)
            #epoch_iterator = self.project_gold(epoch_iterator)

            if self.project_names:
                epoch_iterator = self.project_from_name_wrapper(epoch_iterator)

            batch_iterator = self.iterate_in_batches(epoch_iterator, validate_batches=False)
            for i,batch in enumerate(batch_iterator):
                print("asdf")
                self.preprocessor.process(batch, mode="train")

                assignment_dict = self.model.handle_variable_assignment(batch, mode='train')
                result = self.sess.run([self.optimize_func, model_loss], feed_dict=assignment_dict)
                loss = result[1]

                Static.logger.write("Loss at batch "+str(i) + ": " + str(loss), verbosity_priority=2)

    def predict(self, test_file_iterator):
        example_iterator = test_file_iterator.iterate()
        example_iterator = self.candidate_generator.enrich(example_iterator)

        if self.project_names:
            example_iterator = self.project_from_name_wrapper(example_iterator)
        model_prediction = self.model.get_prediction_graph()
        for example in example_iterator:

            as_batch = {k:[v] for k,v in example.items()}
            self.preprocessor.process(as_batch, mode='predict')

            assignment_dict = self.model.handle_variable_assignment(as_batch, mode='predict')
            predictions = self.sess.run(model_prediction, feed_dict=assignment_dict)

            for i, prediction in enumerate(predictions):
                best_predictions = np.where(prediction[0] > 0.3)[0]
                output = []
                for prediction in best_predictions:
                    output.append(example["neighborhood"].from_index(prediction))

                if self.project_names:
                    output = example["neighborhood"].get_name_connections(output)

                yield output
