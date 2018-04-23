from input_models.abstract_preprocessor import AbstractPreprocessor
from input_models.padded_map.padded_map_input_model import PaddedMapInputModel
import numpy as np


class SentenceToGraphMapPreprocessor(AbstractPreprocessor):

    def process(self, batch_dictionary, mode='train'):
        if self.next_preprocessor is not None:
            self.next_preprocessor.process(batch_dictionary, mode=mode)

        max_sentence_length = batch_dictionary["question_sentence_input_model"].get_max_words_in_batch()
        max_graph_entities = batch_dictionary["neighborhood_input_model"].get_max_graph_entities()

        centroid_map = batch_dictionary["sentence_entity_map"]

        edges = []
        scores = []
        graph_total_vertices = 0
        for i,instance in enumerate(centroid_map):
            true_centroids = batch_dictionary["mentioned_entities"][i]
            true_centroid_indexes = batch_dictionary["neighborhood"][i].centroids
            for centroid in instance:
                matching_true_centroid_index = true_centroid_indexes[true_centroids == centroid[2]][0]

                for j in range(int(centroid[0]), int(centroid[1])+1):
                    edges.append([i*max_sentence_length+j, graph_total_vertices + matching_true_centroid_index])
                    scores.append(float(centroid[3]))
            graph_total_vertices += batch_dictionary["neighborhood"][i].entity_vertices.shape[0]

        edges = np.array(edges) if len(edges) > 0 else np.empty((0,2), dtype=np.int32)
        input_model = PaddedMapInputModel()
        input_model.flat_backward_map = edges[:,0]
        input_model.flat_forward_map = edges[:,1]
        input_model.forward_total_size = graph_total_vertices
        input_model.backward_total_size = len(centroid_map)*max_sentence_length
        input_model.link_scores = scores

        batch_dictionary["sentence_to_neighborhood_map"] = input_model