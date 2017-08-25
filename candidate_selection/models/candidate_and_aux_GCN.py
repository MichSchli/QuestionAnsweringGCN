import numpy as np
import tensorflow as tf

from candidate_selection.models.components.decoders.softmax_decoder import SoftmaxDecoder
from candidate_selection.models.components.extras.embedding_mapper import EmbeddingMapper
from candidate_selection.models.components.extras.target_comparator import TargetComparator
from candidate_selection.models.components.graph_encoders.hypergraph_gcn_propagation_unit import \
    HypergraphGcnPropagationUnit
from candidate_selection.models.components.graph_encoders.vertex_embedding import VertexEmbedding
from candidate_selection.models.lazy_indexer import LazyIndexer
from candidate_selection.tensorflow_hypergraph_representation import TensorflowHypergraphRepresentation
from candidate_selection.tensorflow_variables_holder import TensorflowVariablesHolder
from input_models.hypergraph.hypergraph_preprocessor import HypergraphPreprocessor


class CandidateAndAuxGcnModel:
    """
    A model which computes a max over a GCN representation of the candidate hypergraph.
    """

    dimension = None
    decoder = None
    hypergraph_gcn_propagation_units = None
    facts = None
    variables = None

    embeddings = None

    entity_indexer = None
    relation_indexer = None

    aux_iterator = None

    def __init__(self, facts, aux_iterator, layers=1, dimension=6):
        self.dimension = dimension
        self.entity_dict = {}
        self.facts = facts

        # TODO this is the most retarded thing
        # #AlternativeFacts #Yolo
        aux_facts = facts

        self.aux_iterator = aux_iterator
        self.variables = TensorflowVariablesHolder()

        self.aux_mapper = EmbeddingMapper(self.variables)

        self.hypergraph_batch_preprocessor = HypergraphPreprocessor()
        self.aux_hypergraph_batch_preprocessor = HypergraphPreprocessor()

        self.entity_embedding = VertexEmbedding(facts, self.variables, self.dimension,random=False)
        self.event_embedding = VertexEmbedding(facts, self.variables, self.dimension,random=True)

        self.aux_entity_embedding = VertexEmbedding(aux_facts, self.variables, self.dimension, random=True, variable_prefix="aux")
        self.aux_event_embedding = VertexEmbedding(aux_facts, self.variables, self.dimension, random=True, variable_prefix="aux_event")

        self.hypergraph = TensorflowHypergraphRepresentation(self.variables)
        self.aux_hypergraph = TensorflowHypergraphRepresentation(self.variables, variable_prefix="aux")

        self.target_comparator = TargetComparator(self.variables, variable_prefix="comparison_to_aux")

        self.hypergraph_gcn_propagation_units = [None]*layers
        self.aux_hypergraph_gcn_propagation_units = [None]*layers
        for layer in range(layers):
            self.hypergraph_gcn_propagation_units[layer] = HypergraphGcnPropagationUnit("layer_"+str(layer), aux_facts, self.variables, dimension, self.hypergraph)
            self.aux_hypergraph_gcn_propagation_units[layer] = HypergraphGcnPropagationUnit("aux_layer_"+str(layer), aux_facts, self.variables, dimension, self.aux_hypergraph)

        self.decoder = SoftmaxDecoder(self.variables)

        self.entity_indexer = LazyIndexer()
        self.relation_indexer = LazyIndexer()

    def get_aux_iterators(self):
        return [self.aux_iterator.produce_additional_graphs()]

    def prepare_variables(self, mode='predict'):
        self.aux_mapper.prepare_variables()
        self.hypergraph.prepare_variables()
        self.entity_embedding.prepare_variables()
        self.event_embedding.prepare_variables()

        self.aux_hypergraph.prepare_variables()
        self.aux_entity_embedding.prepare_variables()
        self.aux_event_embedding.prepare_variables()
        self.target_comparator.prepare_variables()

        self.decoder.prepare_variables(mode=mode)

        for hgpu in self.hypergraph_gcn_propagation_units:
            hgpu.prepare_variables()

        for hgpu in self.aux_hypergraph_gcn_propagation_units:
            hgpu.prepare_variables()

    def validate_example(self, batch):
        candidates = batch[0].get_vertices(type="entities")
        target_vertices = batch[-1]

        # For now, eliminate all batches without 100 % overlap
        target_vertices_in_candidates = np.isin(target_vertices, candidates)
        return target_vertices_in_candidates.all()

    def preprocess(self, batch, mode='test'):
        hypergraph_input_model = self.hypergraph_batch_preprocessor.preprocess(batch[0])
        aux_hypergraph_input_model = self.aux_hypergraph_batch_preprocessor.preprocess([k[0] for k in batch[1]])

        transform = np.empty((0,2))
        targets = np.empty(0)
        for i,element in enumerate(batch[1]):
            if len(element[1].items()) > 0:
                transform = np.concatenate((transform, [[self.aux_hypergraph_batch_preprocessor.retrieve_entity_indexes_in_batch(i,k)
                         ,self.hypergraph_batch_preprocessor.retrieve_entity_indexes_in_batch(i,v)] for k,v in element[1].items()]))

            print(element[0].get_vertices("entities"))

            target_v = self.aux_hypergraph_batch_preprocessor.retrieve_entity_indexes_in_batch(i,element[2])
            num_potential_answers = batch[0][i].get_vertices("entities").shape[0]
            targets = np.concatenate((targets, np.repeat(target_v, num_potential_answers)))

        preprocessed = [hypergraph_input_model, aux_hypergraph_input_model, transform, targets]

        if mode == 'train':
            gold_matrix = np.zeros_like(hypergraph_input_model.entity_vertex_matrix, dtype=np.float32)
            shitty_counter = 0
            for i, golds in enumerate(batch[-1]):
                gold_indexes = np.array([self.hypergraph_batch_preprocessor.retrieve_entity_indexes_in_batch(i,gold) for gold in golds])
                gold_matrix[i][gold_indexes - shitty_counter] = 1
                shitty_counter = np.max(hypergraph_input_model.entity_vertex_matrix[i])
            preprocessed += [gold_matrix]

        return preprocessed

    def get_optimizable_parameters(self):
        optimizable_vars = self.entity_embedding.get_optimizable_parameters() #+ self.aux_entity_embedding.get_optimizable_parameters()

        for hgpu in self.hypergraph_gcn_propagation_units:
            optimizable_vars += hgpu.get_optimizable_parameters()

        for hgpu in self.aux_hypergraph_gcn_propagation_units:
            optimizable_vars += hgpu.get_optimizable_parameters()

        return optimizable_vars

    def handle_variable_assignment(self, o_preprocessed_batch, mode="predict"):
        hypergraph_input_model = o_preprocessed_batch[0]
        self.event_embedding.handle_variable_assignment(hypergraph_input_model.n_events)
        self.entity_embedding.handle_variable_assignment(hypergraph_input_model.entity_map)
        self.hypergraph.handle_variable_assignment(hypergraph_input_model)
        self.decoder.handle_variable_assignment(hypergraph_input_model.entity_vertex_matrix, hypergraph_input_model.entity_vertex_slices)

        aux_hypergraph_input_model = o_preprocessed_batch[1]
        self.aux_event_embedding.handle_variable_assignment(aux_hypergraph_input_model.n_events)
        self.aux_entity_embedding.handle_variable_assignment(aux_hypergraph_input_model.n_entities)
        self.aux_hypergraph.handle_variable_assignment(aux_hypergraph_input_model)

        self.aux_mapper.handle_variable_assignment(o_preprocessed_batch[2][:,0],
                                                   o_preprocessed_batch[2][:,1],
                                                   aux_hypergraph_input_model.n_entities,
                                                   hypergraph_input_model.n_entities)

        self.target_comparator.handle_variable_assignment(o_preprocessed_batch[3])

        if mode == 'train':
            self.decoder.assign_gold_variable(o_preprocessed_batch[-1])

        return self.variables.get_assignment_dict()

    def retrieve_entities(self, graph_index, entity_index):
        return [self.hypergraph_batch_preprocessor.retrieve_entity_labels_in_batch(graph_index, entity_index)]

    entity_scores=None

    def get_loss_graph(self, sum_examples=True):
        if self.entity_scores is None:
            self.entity_scores = self.compute_entity_scores()

        return self.decoder.decode_to_loss(self.entity_scores, sum_examples=sum_examples)

    def get_prediction_graph(self):
        if self.entity_scores is None:
            self.entity_scores = self.compute_entity_scores()

        return self.decoder.decode_to_prediction(self.entity_scores)

    def compute_entity_scores(self):
        self.hypergraph.entity_vertex_embeddings = self.entity_embedding.get_representations()
        self.hypergraph.event_vertex_embeddings = self.event_embedding.get_representations()
        self.aux_hypergraph.entity_vertex_embeddings = self.aux_entity_embedding.get_representations()
        self.aux_hypergraph.event_vertex_embeddings = self.aux_event_embedding.get_representations()

        """
        for hgpu, a_hgpu in zip(self.hypergraph_gcn_propagation_units[:-1],
                                self.aux_hypergraph_gcn_propagation_units[:-1]):
            a_hgpu.propagate()
            self.aux_hypergraph.entity_vertex_embeddings = tf.nn.relu(self.aux_hypergraph.entity_vertex_embeddings)
            self.aux_hypergraph.event_vertex_embeddings = tf.nn.relu(self.aux_hypergraph.event_vertex_embeddings)

            self.hypergraph.entity_vertex_embeddings += self.aux_mapper.apply_map(
                self.aux_hypergraph.entity_vertex_embeddings)

            hgpu.propagate()
            self.hypergraph.entity_vertex_embeddings = tf.nn.relu(self.hypergraph.entity_vertex_embeddings)
            self.hypergraph.event_vertex_embeddings = tf.nn.relu(self.hypergraph.event_vertex_embeddings)

        """

        self.aux_hypergraph_gcn_propagation_units[-1].propagate()
        self.aux_hypergraph.entity_vertex_embeddings = tf.nn.relu(self.aux_hypergraph.entity_vertex_embeddings)
        self.aux_hypergraph_gcn_propagation_units[-1].propagate()
        self.aux_hypergraph.entity_vertex_embeddings = tf.nn.relu(self.aux_hypergraph.entity_vertex_embeddings)
        self.aux_hypergraph_gcn_propagation_units[-1].propagate()
        self.aux_hypergraph.entity_vertex_embeddings = tf.nn.relu(self.aux_hypergraph.entity_vertex_embeddings)
        self.aux_hypergraph_gcn_propagation_units[-1].propagate()

        self.hypergraph.entity_vertex_embeddings += self.aux_mapper.apply_map(
            self.aux_hypergraph.entity_vertex_embeddings)

        self.hypergraph_gcn_propagation_units[-1].propagate()
        self.hypergraph.entity_vertex_embeddings = tf.nn.relu(self.hypergraph.entity_vertex_embeddings)
        self.hypergraph_gcn_propagation_units[-1].propagate()
        self.hypergraph.entity_vertex_embeddings = tf.nn.relu(self.hypergraph.entity_vertex_embeddings)
        self.hypergraph_gcn_propagation_units[-1].propagate()
        self.hypergraph.entity_vertex_embeddings = tf.nn.relu(self.hypergraph.entity_vertex_embeddings)
        self.hypergraph_gcn_propagation_units[-1].propagate()

        entity_scores = self.target_comparator.get_comparison_scores(self.aux_hypergraph.entity_vertex_embeddings,
                                                                     self.hypergraph.entity_vertex_embeddings) #tf.reduce_sum(self.hypergraph.entity_vertex_embeddings, 1)

        return entity_scores


