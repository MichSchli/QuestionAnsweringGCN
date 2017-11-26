import tensorflow as tf
import numpy as np
from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


class TensorflowHypergraphRepresentation(AbstractComponent):

    entity_vertex_embeddings = None
    entity_vertex_dimension = None

    event_vertex_embeddings = None
    event_vertex_dimension = None

    variables = None

    def __init__(self, variables, variable_prefix=""):
        self.variables = variables

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def initialize_zero_embeddings(self, dimension):
        self.entity_vertex_embeddings = tf.zeros((self.variables.get_variable(self.variable_prefix + "n_entities"), dimension))
        self.event_vertex_embeddings = tf.zeros((self.variables.get_variable(self.variable_prefix + "n_events"), dimension))

    def update_entity_embeddings(self, embeddings, dimension):
        self.entity_vertex_embeddings = embeddings
        self.entity_vertex_dimension = dimension

    def get_embeddings(self, reference):
        if reference == "entities":
            return self.entity_vertex_embeddings
        elif reference == "events":
            return self.event_vertex_embeddings

    def get_centroid_embeddings(self):
        return tf.nn.embedding_lookup(self.entity_vertex_embeddings, self.variables.get_variable(self.variable_prefix + "centroid_indices"))

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
        return self.variable_prefix + variable_name

    """
    Defining and assigning graph-related variables:
    """

    def prepare_tensorflow_variables(self, mode="train"):
        self.prepare_variable_set(self.variable_prefix + "events_to_entities")
        self.prepare_variable_set(self.variable_prefix + "entities_to_events")
        self.prepare_variable_set(self.variable_prefix + "entities_to_entities")
        self.variables.add_variable(self.variable_prefix + "n_entities", tf.placeholder(tf.int32, name=self.variable_prefix+"n_entities"))
        self.variables.add_variable(self.variable_prefix + "n_events", tf.placeholder(tf.int32, name=self.variable_prefix+"n_events"))

    def prepare_variable_set(self, prefix):
        self.variables.add_variable(prefix+"_edges", tf.placeholder(tf.int32, name=prefix+"_edges"))
        self.variables.add_variable(prefix+"_edge_types", tf.placeholder(tf.int32, name=prefix+"_edge_types"))

    def handle_variable_assignment(self, batch_dict, mode):
        hypergraph_input_model = batch_dict["neighborhood_input_model"]
        self.handle_variable_set_assignment(self.variable_prefix + "events_to_entities", hypergraph_input_model.event_to_entity_edges, hypergraph_input_model.event_to_entity_types)
        self.handle_variable_set_assignment(self.variable_prefix + "entities_to_events", hypergraph_input_model.entity_to_event_edges, hypergraph_input_model.entity_to_event_types)
        self.handle_variable_set_assignment(self.variable_prefix + "entities_to_entities", hypergraph_input_model.entity_to_entity_edges, hypergraph_input_model.entity_to_entity_types)
        self.variables.assign_variable(self.variable_prefix + "n_entities", hypergraph_input_model.n_entities)
        self.variables.assign_variable(self.variable_prefix + "n_events", hypergraph_input_model.n_events)

    def handle_variable_set_assignment(self, prefix, edges, types):
        self.variables.assign_variable(prefix + "_edges", edges)
        self.variables.assign_variable(prefix + "_edge_types", types)