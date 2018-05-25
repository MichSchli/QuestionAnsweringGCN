from indexing.abstract_index import AbstractIndex
import numpy as np


class PureFreebaseRelationIndex(AbstractIndex):

    vector_count = 1250

    def __init__(self, index_cache_name, dimension, keep_space_for_inverse=False):
        AbstractIndex.__init__(self, index_cache_name, dimension)

    def index(self, element):
        return AbstractIndex.index(self, element)