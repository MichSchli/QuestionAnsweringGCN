from indexing.abstract_index import AbstractIndex


class EntityIndex(AbstractIndex):

    def __init__(self, index_cache_name, dimension):
        AbstractIndex.__init__(self, index_cache_name, dimension)
        self.index("<mention_dummy>")
        self.index("<word_dummy>")
        self.index("<dependency_root>")
