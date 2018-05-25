from example_reader.graph_reader.edge_type_utils import EdgeTypeUtils
from example_reader.graph_reader.graph import Graph
import numpy as np


class EmptyGraphReader:

    def __init__(self):
        self.edge_type_utils = EdgeTypeUtils()

    def get_neighborhood_graph(self, mention_entities):
        graph = Graph()
        vertex_types = np.array([[0,0,0,1,0,0] for _ in mention_entities], dtype=np.float32)
        graph.add_vertices(mention_entities, vertex_types)

        label_dict = {k:i for i,k in enumerate(mention_entities)}
        graph.set_label_to_index_map(label_dict)
        graph.set_index_to_name_map({})

        graph.edge_types = [np.array([], dtype=np.int32) for _ in range(self.edge_type_utils.count_types())]

        return graph