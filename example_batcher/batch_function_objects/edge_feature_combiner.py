import numpy as np


class EdgeFeatureCombiner:

    batch = None

    def __init__(self, batch):
        self.batch = batch

    def get_combined_edge_type_indices(self):
        index_lists = [np.copy(example.graph.edges[:,1]) for example in self.batch.examples]

        return np.concatenate(index_lists)

    def get_padded_edge_part_type_matrix(self):
        pad_to = max(example.get_padded_edge_part_type_matrix().shape[1] for example in self.batch.examples)
        index_lists = []

        for i,example in enumerate(self.batch.examples):
            example_bags = example.get_padded_edge_part_type_matrix()
            padding_needed = pad_to - example_bags.shape[1]

            if padding_needed > 0:
                example_bags = np.pad(example_bags, ((0,0), (0,padding_needed)), mode='constant')

            index_lists.append(example_bags)

        return np.concatenate(index_lists)
