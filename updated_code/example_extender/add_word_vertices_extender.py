import numpy as np


class AddWordDummyExtender:

    relation_index = None
    vertex_index = None
    inner = None

    def __init__(self, inner, relation_index, vertex_index):
        self.inner = inner
        self.relation_index = relation_index
        self.vertex_index = vertex_index

    def extend(self, example):
        example = self.inner.extend(example)

        if not example.has_mentions():
            return example

        word_vertices = [None] * example.count_words()
        word_edges = []
        graph_vertex_count = example.count_vertices()

        example.question.dummy_indexes = [None] * len(example.question.words)
        for i, word in enumerate(example.question.words):
            word_vertices[i] = self.vertex_index.index("<word_dummy>")
            example.question.dummy_indexes[i] = graph_vertex_count + i

        for mention in example.mentions:
            for word_index in mention.word_indexes:
                word_edge = [mention.dummy_index,
                             self.relation_index.index("<dummy_to_word>"),
                             graph_vertex_count + word_index]
                word_edges.append(word_edge)

        word_vertices = np.array(word_vertices)
        word_vertex_types = np.array([[0,0,0,1,0] for _ in word_vertices], dtype=np.float32)
        word_edges = np.array(word_edges)

        example.graph.add_vertices(word_vertices, word_vertex_types)
        example.graph.add_edges(word_edges)

        return example
