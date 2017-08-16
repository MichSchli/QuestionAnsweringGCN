import numpy as np

from database_interface.search_filters.prefix_filter import PrefixFilter


class HypergraphPropertyRetriever:

    next_property_retriever = None

    def __init__(self, next_property_retriever):
        self.next_property_retriever = next_property_retriever

    def retrieve_properties(self, vertices, types):
        properties = self.next_property_retriever.retrieve_properties(vertices, types)

        if vertices.shape[0] == 0:
            properties["type"] = np.empty((0,2))
            return properties

        logically_is_hyperedge = self.find_hypergraph_edges(vertices, types, properties)
        properties["type"] = np.stack((vertices[logically_is_hyperedge], ["event" for _ in vertices[logically_is_hyperedge]]), axis=1)
        return properties

    def find_hypergraph_edges(self, entities, types, properties):
        vertex_has_name = np.isin(entities, np.array(list(properties["name"][:,0])))
        vertex_is_not_literal = types != "literal"

        freebase_filter = PrefixFilter("http://rdf.freebase.com/ns/")
        vertex_is_freebase_element = freebase_filter.accepts(entities)
        logically_is_hyperedge = np.logical_and(np.logical_and(np.logical_not(vertex_has_name), vertex_is_not_literal),
                                                vertex_is_freebase_element)
        return logically_is_hyperedge