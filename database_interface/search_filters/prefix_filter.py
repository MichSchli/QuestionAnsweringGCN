import numpy as np


class PrefixFilter:
    """
    Applies a heuristic filter excluding any vertices with a given prefix.
    """

    prefix = None

    def __init__(self, prefix, sign=True):
        self.prefix = prefix

    def accepts(self, elements):
        return np.array([e.startswith(self.prefix) for e in elements])