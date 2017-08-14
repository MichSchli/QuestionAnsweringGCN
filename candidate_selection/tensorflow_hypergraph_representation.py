class TensorflowHypergraphRepresentation:

    entity_vertex_embeddings = None
    entity_vertex_dimension = None

    event_vertex_embeddings = None
    event_vertex_dimension = None

    def update_entity_embeddings(self, embeddings, dimension):
        self.entity_vertex_embeddings = embeddings
        self.entity_vertex_dimension = dimension

    def get_embeddings(self, reference):
        if reference == "entities":
            return self.entity_vertex_embeddings
        elif reference == "events":
            return self.event_vertex_embeddings