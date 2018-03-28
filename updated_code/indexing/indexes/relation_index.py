from indexing.abstract_index import AbstractIndex
import numpy as np


class RelationIndex(AbstractIndex):

    vector_count = 500
    inverse_edge_delimiter = 250
