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
    example_processor = None
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

    def set_example_processor(self, example_processor):
        self.example_processor = example_processor

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

    def train(self, train_file_iterator, start_epoch=0, epochs=None):
        if epochs is None:
            epochs = self.epochs

        model_prediction = self.model.get_prediction_graph()

        for epoch in range(start_epoch, start_epoch+epochs):
            Static.logger.write("Starting epoch " + str(epoch), "training", "iteration_messages")
            epoch_iterator = train_file_iterator.iterate(shuffle=True)
            epoch_iterator = self.candidate_generator.enrich(epoch_iterator)
            #epoch_iterator = self.split_graphs(epoch_iterator)

            epoch_iterator = self.example_processor.process_stream(epoch_iterator, mode="train")

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
        #example_iterator = self.split_graphs(example_iterator)

        example_iterator = self.example_processor.process_stream(example_iterator, mode="predict")

        model_prediction = self.model.get_prediction_graph()
        edge_gates = self.model.get_edge_gates()
        batch_iterator = self.iterate_in_batches(example_iterator, validate_batches=False)
        for j,batch in enumerate(batch_iterator):
            print("batch "+str(j)+":\n - - - - - - - ")

            self.preprocessor.process(batch, mode='predict')

            assignment_dict = self.model.handle_variable_assignment(batch, mode='predict')
            predictions, gates = self.sess.run([model_prediction, edge_gates], feed_dict=assignment_dict)
            edge_counts = [0,0,0]

            for i, prediction in enumerate(predictions):
                l = list(sorted(enumerate(prediction[0]), key=lambda x: x[1], reverse=True))

                #formatted_gate_information, edge_counts = self.format_gate_information(batch["neighborhood"][i], batch["gold_entities"][i], prediction[0], gates, edge_counts)
                #self.print_formatted_gate_information(formatted_gate_information, batch["sentence"][i])

                for index,prob in l[:5]:
                    if prob > 0.5:
                        print(batch["neighborhood"][i].get_paths_to_neighboring_centroid(index))

                yield [(batch["neighborhood"][i].from_index_with_names(index),prob) for index,prob in l]

    def print_formatted_gate_information(self, formatted_gate_information, sentence):
        print(" ".join([w[1] for w in sentence]))
        print("-----")

        for edge in formatted_gate_information[0]:
            print("\t".join(edge))
        print("-----")
        for edge in formatted_gate_information[1]:
            print("\t".join(edge))
        print("-----")
        for edge in formatted_gate_information[2]:
            print("\t".join(edge))

        print("\n")


    def format_gate_information(self, hypergraph, gold, predictions, all_gates_in_batch, edge_counts):
        layer = 0
        centroid_scores = hypergraph.centroid_scores
        all_formated_edges = []

        formatted_gate_information = self.get_formatted_en_to_ev(all_gates_in_batch, centroid_scores, edge_counts,
                                                                 gold, hypergraph, layer,
                                                                 predictions)
        all_formated_edges.append(formatted_gate_information)


        formatted_gate_information = self.get_formatted_ev_to_en(all_gates_in_batch, centroid_scores, edge_counts,
                                                                 gold, hypergraph, layer,
                                                                 predictions)
        all_formated_edges.append(formatted_gate_information)


        formatted_gate_information = self.get_formatted_en_to_en(all_gates_in_batch, centroid_scores, edge_counts,
                                                                 gold, hypergraph, layer,
                                                                 predictions)
        all_formated_edges.append(formatted_gate_information)

        return all_formated_edges, edge_counts

    def get_formatted_en_to_ev(self, all_gates_in_batch, centroid_scores, edge_counts, gold, hypergraph,
                               layer, predictions):
        en_to_ev_edges = hypergraph.entity_to_event_edges
        n_en_to_ev_edges = hypergraph.entity_to_event_edges.shape[0]

        en_to_ev_gates = all_gates_in_batch[layer][0][edge_counts[0]:edge_counts[0] + n_en_to_ev_edges]
        en_to_ev_invert_gates = all_gates_in_batch[layer][4][edge_counts[0]:edge_counts[0] + n_en_to_ev_edges]
        formatted_gate_information = [["_"] * 12 for _ in range(n_en_to_ev_edges)]
        pointer = 0
        for edge, gate, invert_gate in zip(en_to_ev_edges, en_to_ev_gates, en_to_ev_invert_gates):
            formatted_gate_information[pointer][0] = hypergraph.from_index_with_names(edge[0])
            formatted_gate_information[pointer][1] = hypergraph.relation_map[edge[1]]
            formatted_gate_information[pointer][2] = "cvt|id=" + str(hypergraph.event_vertices[edge[2]])
            formatted_gate_information[pointer][3] = str(gate[0])
            formatted_gate_information[pointer][4] = str(invert_gate[0])

            if edge[0] in gold:
                formatted_gate_information[pointer][6] = "subject_is_gold"

            if centroid_scores[edge[0]] > 0:
                formatted_gate_information[pointer][8] = "subject_is_centroid|score=" + str(centroid_scores[edge[0]])

            formatted_gate_information[pointer][10] = str(predictions[edge[0]])

            pointer += 1
        edge_counts[0] += n_en_to_ev_edges
        return formatted_gate_information

    def get_formatted_ev_to_en(self, all_gates_in_batch, centroid_scores, edge_counts, gold, hypergraph,
                               layer, predictions):
        edges = hypergraph.event_to_entity_edges
        n_edges = hypergraph.event_to_entity_edges.shape[0]

        gates = all_gates_in_batch[layer][1][edge_counts[1]:edge_counts[1] + n_edges]
        invert_gates = all_gates_in_batch[layer][3][edge_counts[1]:edge_counts[1] + n_edges]
        formatted_gate_information = [["_"] * 12 for _ in range(n_edges)]
        pointer = 0
        for edge, gate, invert_gate in zip(edges, gates, invert_gates):
            formatted_gate_information[pointer][0] = "cvt|id=" + str(hypergraph.event_vertices[edge[0]])
            formatted_gate_information[pointer][1] = hypergraph.relation_map[edge[1]]
            formatted_gate_information[pointer][2] = hypergraph.from_index_with_names(edge[2])
            formatted_gate_information[pointer][3] = str(gate[0])
            formatted_gate_information[pointer][4] = str(invert_gate[0])

            if edge[2] in gold:
                formatted_gate_information[pointer][7] = "object_is_gold"

            if centroid_scores[edge[2]] > 0:
                formatted_gate_information[pointer][9] = "object_is_centroid|score=" + str(centroid_scores[edge[2]])

            formatted_gate_information[pointer][11] = str(predictions[edge[2]])

            pointer += 1
        edge_counts[1] += n_edges
        return formatted_gate_information

    def get_formatted_en_to_en(self, all_gates_in_batch, centroid_scores, edge_counts, gold, hypergraph,
                               layer, predictions):
        edges = hypergraph.entity_to_entity_edges
        n_edges = hypergraph.entity_to_entity_edges.shape[0]

        gates = all_gates_in_batch[layer][2][edge_counts[2]:edge_counts[2] + n_edges]
        invert_gates = all_gates_in_batch[layer][5][edge_counts[2]:edge_counts[2] + n_edges]
        formatted_gate_information = [["_"] * 12 for _ in range(n_edges)]
        pointer = 0
        for edge, gate, invert_gate in zip(edges, gates, invert_gates):
            formatted_gate_information[pointer][0] = hypergraph.from_index_with_names(edge[0])
            formatted_gate_information[pointer][1] = hypergraph.relation_map[edge[1]]
            formatted_gate_information[pointer][2] = hypergraph.from_index_with_names(edge[2])
            formatted_gate_information[pointer][3] = str(gate[0])
            formatted_gate_information[pointer][4] = str(invert_gate[0])

            if edge[0] in gold:
                formatted_gate_information[pointer][6] = "subject_is_gold"

            if edge[2] in gold:
                formatted_gate_information[pointer][7] = "object_is_gold"

            if centroid_scores[edge[0]] > 0:
                formatted_gate_information[pointer][8] = "subject_is_centroid|score=" + str(centroid_scores[edge[0]])

            if centroid_scores[edge[2]] > 0:
                formatted_gate_information[pointer][9] = "object_is_centroid|score=" + str(centroid_scores[edge[2]])

            formatted_gate_information[pointer][10] = str(predictions[edge[0]])
            formatted_gate_information[pointer][11] = str(predictions[edge[2]])

            pointer += 1
        edge_counts[2] += n_edges
        return formatted_gate_information