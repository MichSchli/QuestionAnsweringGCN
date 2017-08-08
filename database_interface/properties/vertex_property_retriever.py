from database_interface.search_filters.prefix_filter import PrefixFilter


class VertexPropertyRetriever:
    data_interface = None
    filter = None

    def __init__(self, data_interface):
        self.data_interface = data_interface
        self.filter = PrefixFilter("http://rdf.freebase.com/ns/")

    def retrieve_properties(self, vertices):
        name_properties = self.data_interface.get_property(vertices, "ns:type.object.name")
        return {"name": name_properties}