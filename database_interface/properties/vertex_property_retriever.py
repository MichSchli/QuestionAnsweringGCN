from database_interface.search_filters.prefix_filter import PrefixFilter
import numpy as np


class VertexPropertyRetriever:
    data_interface = None
    filter = None

    def __init__(self, data_interface):
        self.data_interface = data_interface
        self.filter = PrefixFilter("http://rdf.freebase.com/ns/")

    def retrieve_properties(self, vertices, types):
        if vertices.shape[0] == 0:
            return {"name": np.empty((0,2))}
        #print(self.filter.accepts(vertices))
        #print(vertices)
        pass_vertices = vertices[self.filter.accepts(vertices)]
        #print(pass_vertices)
        #print(vertices)
        name_properties = self.data_interface.get_property(pass_vertices, "ns:type.object.name")
        return {"name": name_properties}
