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
        graph_total_vertices = 0
        for i,instance in enumerate(centroid_map):
            for centroid in instance:
                centroid_index_in_graph = batch_dictionary["neighborhood"][i].to_index(centroid[2])
                for j in range(int(centroid[0]), int(centroid[1])+1):
                    edges.append([i*max_sentence_length+j, graph_total_vertices + centroid_index_in_graph])
            graph_total_vertices += batch_dictionary["neighborhood"][i].entity_vertices.shape[0]

        edges = np.array(edges)
        input_model = PaddedMapInputModel()
        input_model.flat_backward_map = edges[:,0]
        input_model.flat_forward_map = edges[:,1]
        input_model.forward_total_size = graph_total_vertices
        input_model.backward_total_size = len(centroid_map)*max_sentence_length

        batch_dictionary["sentence_to_neighborhood_map"] = input_model