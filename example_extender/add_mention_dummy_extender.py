import numpy as np

from example_reader.graph_reader.edge_type_utils import EdgeTypeUtils


class AddMentionDummyExtender:

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

        mention_vertices = [None]*len(example.mentions)
        mention_edges = [None]*len(example.mentions)
        graph_vertex_count = example.count_vertices()

        for i, mention in enumerate(example.mentions):
            mention_vertices[i] = self.vertex_index.index("<mention_dummy>")
            mention.dummy_index = graph_vertex_count + i
            mention_edges[i] = [mention.dummy_index,
                                self.relation_index.index("<dummy_to_mention>"),
                                mention.entity_index]

        mention_vertices = np.array(mention_vertices)
        mention_vertex_types = np.array([[0, 0, 1, 0, 0, 0] for _ in mention_vertices], dtype=np.float32)
        mention_edges = np.array(mention_edges)

        example.graph.add_vertices(mention_vertices, mention_vertex_types)
        example.graph.edge_types[self.edge_type_utils.index_of("mention_dummy")] = np.arange(len(mention_edges), dtype=np.int32) + example.graph.edges.shape[0]
        example.graph.add_edges(mention_edges)

        return example
