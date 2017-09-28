import numpy as np


class InfixFilter:
    """
    Applies a heuristic filter excluding any vertices without a given infix.
    """

    prefix = None
    position = None

    def __init__(self, infix, position):
        self.infix = infix
        self.position = position

    def accepts(self, elements):
        acceptance = np.array([e[self.position] == self.infix for e in elements])

        #for v1, v2 in zip(elements, acceptance):
        #    print("accept " + str(v1)+": "+str(v2))

        return acceptance
