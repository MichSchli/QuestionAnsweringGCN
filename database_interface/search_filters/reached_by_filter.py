import numpy as np


class PrefixFilter:
    """
    Applies a special heuristic filter excluding any vertices not reached from a specific pattern.
    """

    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix

    def accepts(self, edge_type, vertex):
        pass
