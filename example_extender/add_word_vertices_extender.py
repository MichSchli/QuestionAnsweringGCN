import numpy as np

from example_reader.graph_reader.edge_type_utils import EdgeTypeUtils


class AddWordDummyExtender:

    relation_index = None
    vertex_index = None
    inner = None

    def __init__(self, inner, relation_index, vertex_index):
        self.inner = inner
        self.relation_index = relation_index
        self.vertex_index = vertex_index
        self.edge_type_utils = EdgeTypeUtils()

    def extend(self, example):
        example = self.inner.extend(example)

        if not example.has_mentions():
            return example

        word_vertices = [None] * example.count_words()
        word_edges = []
        graph_vertex_count = example.count_vertices()

        sentence_dummy_index = graph_vertex_count
        sentence_node = self.vertex_index.index("<sentence_dummy>")
        example.graph.add_vertices(np.array([sentence_node]), np.array([[0, 0, 0, 0, 0, 1]], dtype=np.int32))
        example.graph.sentence_vertex_index = sentence_dummy_index
        graph_vertex_count += 1

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
                word_edge_2 = [sentence_dummy_index,
                             self.relation_index.index("<sentence_to_word>"),
                             graph_vertex_count + word_index]
                word_edges.append(word_edge_2)

        word_vertices = np.array(word_vertices)
        word_vertex_types = np.array([[0,0,0,1,0,0] for _ in word_vertices], dtype=np.float32)
        word_edges = np.array(word_edges)

        example.graph.edge_types[self.edge_type_utils.index_of("mention_dummy")] = np.concatenate((example.graph.edge_types[self.edge_type_utils.index_of("mention_dummy")],
                                                                                                   np.arange(word_edges.shape[0]/2, dtype=np.int32) * 2 + example.graph.edges.shape[0]))
        example.graph.edge_types[self.edge_type_utils.index_of("sentence_dummy")] = np.arange(word_edges.shape[0]/2, dtype=np.int32) * 2 + 1 + example.graph.edges.shape[0]

        example.graph.add_vertices(word_vertices, word_vertex_types)
        example.graph.add_edges(word_edges)

        return example
