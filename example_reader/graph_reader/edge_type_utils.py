import numpy as np

class EdgeTypeUtils:

    index_dict = {"entity_to_entity" : 0,
                  "event_to_entity" : 1,
                  "entity_to_event" : 2,
                  "word_sequence" : 3,
                  "mention_dummy" : 4,
                  "sentence_dummy" : 5,
                  "dependency" : 6}

    def get_edge_type_array(self, shape, label):
        index = self.index_of(label)

        return np.ones(shape, dtype=np.int32) * index

    def index_of(self, label):
        return self.index_dict[label]

    def count_types(self):
        return len(self.index_dict)
