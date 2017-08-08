class VertexPropertyRetriever:
    data_interface = None

    def __init__(self, data_interface):
        self.data_interface = data_interface

    def retrieve_properties(self, vertices):
        name_properties = self.data_interface.get_property(vertices, "ns:type.object.name")
        return {"name": name_properties}