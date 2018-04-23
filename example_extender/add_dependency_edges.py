import numpy as np


class AddDependencyEdgeExtender:

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

        root_node_index = example.count_vertices()
        root_node = self.vertex_index.index("<dependency_root>")
        example.graph.add_vertices(np.array([root_node]), np.array([[0,0,0,0,1,0]], dtype=np.int32))

        dep_edges = [None] * len(example.question.dummy_indexes)
        i = 0
        for index, dep_type, dep_head in zip(example.question.dummy_indexes,
                                             example.question.dep_types,
                                             example.question.dep_heads):
            dep_head_idx = example.question.dummy_indexes[dep_head] if dep_head > 0 else root_node_index
            dep_type_idx = self.relation_index.index("<dep:"+dep_type+">")
            dep_edges[i] = [index, dep_type_idx, dep_head_idx]

            i += 1

        dep_edges = np.array(dep_edges)
        example.graph.add_edges(dep_edges)

        return example
