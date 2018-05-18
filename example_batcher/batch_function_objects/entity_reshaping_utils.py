import numpy as np

class EntityReshapingUtils:

    def __init__(self, batch):
        self.batch = batch

    def get_padded_entity_indexes(self):
        max_entity_count = max(example.count_entities() for example in self.batch.examples)
        matrix = np.ones((self.batch.count_examples(), max_entity_count), dtype=np.int32) * -1
        running_count = 0
        for i,example in enumerate(self.batch.examples):
            matrix[i][:example.count_entities()] = np.arange(example.count_entities(), dtype=np.int32) + running_count
            running_count += example.count_entities()

        return matrix