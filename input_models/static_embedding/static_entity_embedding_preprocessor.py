from input_models.abstract_preprocessor import AbstractPreprocessor
import numpy as np


class StaticEntityEmbeddingPreprocessor(AbstractPreprocessor):

    graph_string = None
    dimension = None
    indexer = None

    def __init__(self, indexer, graph_string, next_preprocessor):
        AbstractPreprocessor.__init__(self, next_preprocessor)
        self.graph_string = graph_string
        self.indexer = indexer
        self.dimension = self.indexer.get_dimension()

    def retrieve_embedding(self, entity):
        return self.indexer.retrieve_vector(entity)

    def process(self, batch_dictionary, mode="train"):
        if self.next_preprocessor is not None:
            self.next_preprocessor.process(batch_dictionary, mode=mode)

        hypergraph_entities = batch_dictionary[self.graph_string].entity_map
        batch_dictionary[self.graph_string].entity_embeddings = np.empty((hypergraph_entities.shape[0], self.dimension), dtype=np.float32)

        for i,entity in enumerate(hypergraph_entities):
            batch_dictionary[self.graph_string].entity_embeddings[i] = self.retrieve_embedding(entity)