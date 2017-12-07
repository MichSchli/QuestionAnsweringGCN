from experiment_construction.index_construction.indexes.lazy_indexer import LazyIndexer


class FreebaseRelationIndexer:

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
                relation_name = parts[0]
                count = int(parts[1])

                if count > cutoff_count:
                    self.index_single_element(relation_name)

        self.inner_indexer.freeze()

    def index_single_element(self, element):
        return self.inner_indexer.index_single_element(element)

    def get_dimension(self):
        return self.inner_indexer.get_dimension()

    def retrieve_vector(self, index):
        return self.inner_indexer.retrieve_vector(index)

    def get_all_vectors(self):
        return self.inner_indexer.get_all_vectors()

    def index(self, elements):
        return self.inner_indexer.index(elements)
