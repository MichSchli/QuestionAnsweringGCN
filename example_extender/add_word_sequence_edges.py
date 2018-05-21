import numpy as np

from example_reader.graph_reader.edge_type_utils import EdgeTypeUtils


class AddWordSequenceEdgeExtender:

    inner = None

    def __init__(self, inner, relation_index):
        self.inner = inner
        self.relation_index = relation_index
        self.edge_type_utils = EdgeTypeUtils()

    def extend(self, example):
        example = self.inner.extend(example)

        if not example.has_mentions():
            return example

        word_edges = [None]*(len(example.question.dummy_indexes)-1)

        for i in range(1,len(example.question.dummy_indexes)):
            word_edges[i-1] = [example.question.dummy_indexes[i-1],
                             self.relation_index.index("<word_to_word>"),
                             example.question.dummy_indexes[i]]

        word_edges = np.array(word_edges)
        example.graph.edge_types[self.edge_type_utils.index_of("word_sequence")] = np.arange(word_edges.shape[0], dtype=np.int32) + example.graph.edges.shape[0]
        example.graph.add_edges(word_edges)

        return example
