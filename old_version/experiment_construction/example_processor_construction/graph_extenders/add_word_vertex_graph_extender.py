from experiment_construction.example_processor_construction.abstract_example_processor import AbstractExampleProcessor
import numpy as np


class AddWordVertexGraphExtender(AbstractExampleProcessor):

    """
    Adds vertices corresponding to words to the graph, and adds edges to either mentions or dummy span vertices.
    """

    word_indexer = None

    def __init__(self, next, word_indexer):
        AbstractExampleProcessor.__init__(self, next)
        self.word_indexer = word_indexer

    def process_example(self, example, mode="train"):
        self.add_words(example['neighborhood'], example['sentence'])
        self.add_mention_events(example['neighborhood'], example['sentence_entity_map'])
        return True

    def add_words(self, graph, sentence):
        word_vertices = np.array([w[1] for w in sentence])
        indexed_word_vertices = self.word_indexer.index(word_vertices)
        graph.set_word_vertices(indexed_word_vertices, word_vertices)

    def add_mention_events(self, graph, sentence_entity_map):
        if sentence_entity_map.shape[0] == 0:
            graph.word_to_event_edges = np.empty((0,3), dtype=np.int32)
            return

        #TODO: FIX TYPING ON EDGES

        n_events = graph.event_vertices.shape[0]
        mention_events = [i+n_events for i in range(sentence_entity_map.shape[0])]

        word_to_event_edges = []
        event_to_entity_edges = []

        for i,mention in enumerate(sentence_entity_map):
            for word_index in range(int(mention[0]), int(mention[1])+1):
                word_to_event_edges.append([word_index, 0, i+n_events])
            event_to_entity_edges.append([i+n_events, 1, graph.to_index(mention[2])[0]])

        graph.event_vertices = np.concatenate((graph.event_vertices, mention_events))
        graph.event_to_entity_edges = np.concatenate((graph.event_to_entity_edges, event_to_entity_edges))

        type_bow_dim = graph.event_to_entity_relation_bags.shape[1] if graph.event_to_entity_relation_bags.shape[1] > 0 else 1
        type_bows = np.zeros((len(event_to_entity_edges), type_bow_dim), dtype=np.int32)

        graph.event_to_entity_relation_bags = np.concatenate((graph.event_to_entity_relation_bags, type_bows)) if graph.event_to_entity_relation_bags.shape[0] > 0 else type_bows

        graph.word_to_event_edges = np.array(word_to_event_edges)