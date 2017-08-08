import numpy as np


class PrefixFilter:
    """
    Applies a heuristic filter excluding any vertices with a given prefix.
    """

    prefix = None

    def __init__(self, prefix, sign=True):
        self.prefix = prefix

    def accepts(self, elements):
        acceptance = np.array([e.startswith(self.prefix) for e in elements])

        #for v1, v2 in zip(elements, acceptance):
        #    print("accept " + str(v1)+": "+str(v2))

        return acceptance
