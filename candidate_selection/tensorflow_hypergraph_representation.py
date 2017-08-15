import tensorflow as tf


class TensorflowHypergraphRepresentation:

    entity_vertex_embeddings = None
    entity_vertex_dimension = None

    event_vertex_embeddings = None
    event_vertex_dimension = None

    variables = None

    def __init__(self, variables):
        self.variables = variables

    def update_entity_embeddings(self, embeddings, dimension):
        self.entity_vertex_embeddings = embeddings
        self.entity_vertex_dimension = dimension

    def get_embeddings(self, reference):
        if reference == "entities":
            return self.entity_vertex_embeddings
        elif reference == "events":
            return self.event_vertex_embeddings

    def get_edges(self, senders="events", receivers="entities", inverse_edges=False):
        variable_name = self.compute_variable_name("edges", senders, receivers, inverse_edges)
        variable = self.variables.get_variable(variable_name)
        if not inverse_edges:
            return variable[:,0], variable[:,1]
        else:
            return variable[:,1], variable[:,0]

    def get_edge_types(self, senders="events", receivers="entities", inverse_edges=False):
        variable_name = self.compute_variable_name("edge_types", senders, receivers, inverse_edges)

        return self.variables.get_variable(variable_name)

    def compute_variable_name(self, suffix, senders, receivers, inverse_edges):
        if not inverse_edges:
            variable_name = senders + "_to_" + receivers + "_" + suffix
        else:
            variable_name = receivers + "_to_" + senders + "_" + suffix
        return variable_name

    """
    Defining and assigning graph-related variables:
    """

    def prepare_variables(self):
        self.prepare_variable_set("events_to_entities")
        self.prepare_variable_set("entities_to_events")
        self.prepare_variable_set("entities_to_entities")

    def prepare_variable_set(self, prefix):
        self.variables.add_variable(prefix+"_edges", tf.placeholder(tf.int32, name=prefix+"_edges"))
        self.variables.add_variable(prefix+"_edge_types", tf.placeholder(tf.int32, name=prefix+"_edge_types"))

    def handle_variable_assignment(self, edge_and_type_sets):
        self.handle_variable_set_assignment("events_to_entities", edge_and_type_sets[0], edge_and_type_sets[1])
        self.handle_variable_set_assignment("entities_to_events", edge_and_type_sets[2], edge_and_type_sets[3])
        self.handle_variable_set_assignment("entities_to_entities", edge_and_type_sets[4], edge_and_type_sets[5])

    def handle_variable_set_assignment(self, prefix, edges, types):
        self.variables.assign_variable(prefix + "_edges", edges)
        self.variables.assign_variable(prefix + "_edge_types", types)