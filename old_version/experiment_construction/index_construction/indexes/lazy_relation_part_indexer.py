from experiment_construction.index_construction.indexes.lazy_indexer import LazyIndexer
import numpy as np

class LazyRelationPartIndexer:

    inner_indexer = None

    def __init__(self, vocabulary_shape):
        self.inner_indexer = LazyIndexer(vocabulary_shape)

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
