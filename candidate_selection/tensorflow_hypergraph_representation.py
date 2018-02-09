import tensorflow as tf
import numpy as np
from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


class TensorflowHypergraphRepresentation(AbstractComponent):

    entity_vertex_embeddings = None
    entity_vertex_dimension = None

    event_vertex_embeddings = None
    event_vertex_dimension = None

    variables = None

    def __init__(self, variables, variable_prefix="", edge_dropout_rate=0.0):
        self.variables = variables

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

        self.edge_dropout_rate = edge_dropout_rate

    def initialize_zero_embeddings(self, dimension):
        self.entity_vertex_embeddings = tf.zeros((self.variables.get_variable(self.variable_prefix + "n_entities"), dimension))
        self.event_vertex_embeddings = tf.zeros((self.variables.get_variable(self.variable_prefix + "n_events"), dimension))

    def initialize_with_centroid_scores(self):
        self.entity_vertex_embeddings = tf.expand_dims(self.variables.get_variable(self.variable_prefix + "centroid_scores"), -1)

        #TODO: Receiver features needs this, but it is zero. We need to fix embeddings for these
        self.event_vertex_embeddings = tf.zeros((self.variables.get_variable(self.variable_prefix + "n_events"), 100))

    def update_entity_embeddings(self, embeddings, dimension):
        self.entity_vertex_embeddings = embeddings
        self.entity_vertex_dimension = dimension

    def update_event_embeddings(self, embeddings, dimension):
        self.event_vertex_embeddings = embeddings
        self.event_vertex_dimension = dimension

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

    def distribute_to_edges(self, vectors_by_batch, senders="events", receivers="entities", inverse_edges=False):
        variable_name = self.compute_variable_name("edge_to_batch_indices", senders, receivers, inverse_edges)
        indices = self.variables.get_variable(variable_name)
        distributed_vectors = tf.nn.embedding_lookup(vectors_by_batch, indices)

        return distributed_vectors

    def get_edge_types(self, senders="events", receivers="entities", inverse_edges=False):
        variable_name = self.compute_variable_name("edge_types", senders, receivers, inverse_edges)

        return self.variables.get_variable(variable_name)

    def compute_variable_name(self, suffix, senders, receivers, inverse_edges):
        if not inverse_edges:
            variable_name = senders + "_to_" + receivers + "_" + suffix
        else:
            variable_name = receivers + "_to_" + senders + "_" + suffix
        return self.variable_prefix + variable_name

    def get_event_scores(self):
        return self.variables.get_variable(self.variable_prefix + "event_scores")

    def get_vertex_scores(self):
        return self.variables.get_variable(self.variable_prefix + "vertex_scores")

    """
    Defining and assigning graph-related variables:
    """

    def prepare_tensorflow_variables(self, mode="train"):
        self.prepare_variable_set(self.variable_prefix + "events_to_entities")
        self.prepare_variable_set(self.variable_prefix + "entities_to_events")
        self.prepare_variable_set(self.variable_prefix + "entities_to_entities")
        self.variables.add_variable(self.variable_prefix + "n_entities", tf.placeholder(tf.int32, name=self.variable_prefix+"n_entities"))
        self.variables.add_variable(self.variable_prefix + "n_events", tf.placeholder(tf.int32, name=self.variable_prefix+"n_events"))
        self.variables.add_variable(self.variable_prefix + "vertex_scores", tf.placeholder(tf.float32, name=self.variable_prefix + "vertex_scores"))
        self.variables.add_variable(self.variable_prefix + "event_scores", tf.placeholder(tf.float32, name=self.variable_prefix + "event_scores"))
        self.variables.add_variable(self.variable_prefix + "centroid_scores", tf.placeholder(tf.float32, name=self.variable_prefix + "vertex_scores"))

    def prepare_variable_set(self, prefix):
        self.variables.add_variable(prefix+"_edges", tf.placeholder(tf.int32, name=prefix+"_edges"))
        self.variables.add_variable(prefix+"_edge_types", tf.placeholder(tf.int32, name=prefix+"_edge_types"))
        self.variables.add_variable(prefix+"_edge_to_batch_indices", tf.placeholder(tf.int32, name=prefix+"_edge_to_batch_indices"))

    def handle_variable_assignment(self, batch_dict, mode):
        hypergraph_input_model = batch_dict["neighborhood_input_model"]

        edges, types = self.edge_dropout(hypergraph_input_model.event_to_entity_edges, hypergraph_input_model.event_to_entity_types, mode)
        self.handle_variable_set_assignment(self.variable_prefix + "events_to_entities", edges, types, hypergraph_input_model.ev_to_en_to_batch_map)

        edges, types = self.edge_dropout(hypergraph_input_model.entity_to_event_edges, hypergraph_input_model.entity_to_event_types, mode)
        self.handle_variable_set_assignment(self.variable_prefix + "entities_to_events", edges, types, hypergraph_input_model.en_to_ev_to_batch_map)

        edges, types = self.edge_dropout(hypergraph_input_model.entity_to_entity_edges, hypergraph_input_model.entity_to_entity_types, mode)
        self.handle_variable_set_assignment(self.variable_prefix + "entities_to_entities", edges, types, hypergraph_input_model.en_to_en_to_batch_map)

        self.variables.assign_variable(self.variable_prefix + "n_entities", hypergraph_input_model.n_entities)
        self.variables.assign_variable(self.variable_prefix + "n_events", hypergraph_input_model.n_events)
        self.variables.assign_variable(self.variable_prefix + "vertex_scores", hypergraph_input_model.entity_scores)
        self.variables.assign_variable(self.variable_prefix + "event_scores", hypergraph_input_model.event_scores)
        self.variables.assign_variable(self.variable_prefix + "centroid_scores", hypergraph_input_model.centroid_scores)

    def edge_dropout(self, edges, types, mode):
        dropout_rate = self.edge_dropout_rate
        keep_rate = 1 - dropout_rate
        size = edges.shape[0]

        if size == 0 or dropout_rate == 0 or mode != "train":
            return edges, types

        sample = np.random.choice(np.arange(size), size=int(keep_rate * size), replace=False)
        return edges[sample], types[sample]

    def handle_variable_set_assignment(self, prefix, edges, types, batch_indices):
        self.variables.assign_variable(prefix + "_edges", edges)
        self.variables.assign_variable(prefix + "_edge_types", types)
        self.variables.assign_variable(prefix + "_edge_to_batch_indices", batch_indices)