class AbstractIndexer:

    inner = None
    is_frozen = None

    def __init__(self, inner):
        self.inner = inner
        self.is_frozen = False

    def freeze(self):
        self.is_frozen = True

    def index_single_element(self, element):
        return self.inner.index_single_element(element)

    def get_dimension(self):
        return self.inner.get_dimension()

    def retrieve_vector(self, index):
        return self.inner.retrieve_vector(index)

    def get_all_vectors(self):
        return self.inner.get_all_vectors()

    def index(self, elements):
        return self.inner.index(elements)