from experiment_construction.index_construction.indexes.lazy_indexer import LazyIndexer
import numpy as np

class FreebaseRelationPartIndexer:

    inner_indexer = None

    def __init__(self, vocabulary_shape, cutoff_count):
        self.inner_indexer = LazyIndexer(vocabulary_shape)
        self.load_file(cutoff_count)

    def load_file(self, cutoff_count):
        file = "/home/mschlic1/GCNQA/data/webquestions/filtered_edge_count.txt"

        for line in open(file):
            line = line.strip()
            if line:
                parts = line.split("\t")
                count = int(parts[1])

                if count > cutoff_count:
                    relation_name = parts[0]
                    self.index_single_element(relation_name)

        self.inner_indexer.freeze()

    def index_single_element(self, element):
        true_relation_name = element.split("/")[-1]
        bag = np.unique(true_relation_name.replace(".", "_").split("_"))
        indexed_bag = self.inner_indexer.index(bag)
        return indexed_bag

    def get_dimension(self):
        return self.inner_indexer.get_dimension()

    def retrieve_vector(self, index):
        return self.inner_indexer.retrieve_vector(index)

    def get_all_vectors(self):
        return self.inner_indexer.get_all_vectors()

    def index(self, elements):
        return [self.index_single_element(element) for element in elements]
