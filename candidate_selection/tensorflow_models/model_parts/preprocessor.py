from input_models.hypergraph.hypergraph_preprocessor import HypergraphPreprocessor
from input_models.mask.mask_preprocessor import LookupMaskPreprocessor
from input_models.sentences.sentence_preprocessor import SentencePreprocessor
from input_models.static_embedding.static_entity_embedding_preprocessor import StaticEntityEmbeddingPreprocessor


class PreprocessorPart:

    preprocessor_types = None
    stack = None
    hypergraph_batch_preprocessor = None

    def __init__(self, preprocessor_types, word_indexer, entity_indexer, relation_indexer):
        self.preprocessor_types = preprocessor_types
        self.word_indexer = word_indexer
        self.entity_indexer = entity_indexer
        self.relation_indexer = relation_indexer

    def initialize_preprocessor(self, preprocessor_type):
        if preprocessor_type == "hypergraph":
            self.hypergraph_batch_preprocessor = HypergraphPreprocessor(self.entity_indexer, self.relation_indexer,
                                                                        "neighborhood", "neighborhood_input_model",
                                                                        self.stack)
            self.stack = self.hypergraph_batch_preprocessor
        elif preprocessor_type == "gold":
            self.stack = LookupMaskPreprocessor("neighborhood_input_model", "entity_vertex_matrix", "gold_entities",
                                                   "gold_mask", self.stack)
        elif preprocessor_type == "sentence":
            self.stack = SentencePreprocessor(self.word_indexer, "sentence", "question_sentence_input_model",
                                                     self.stack)
        elif preprocessor_type == "static_entity_embeddings":
            self.stack = StaticEntityEmbeddingPreprocessor(self.entity_indexer, "neighborhood_input_model", self.stack)

    def initialize_all_preprocessors(self):
        for preprocessor_type in self.preprocessor_types:
            self.initialize_preprocessor(preprocessor_type)

    def process(self, batch):
        return self.stack.process(batch)

    def retrieve_entities(self, graph_index, entity_index):
        return [self.hypergraph_batch_preprocessor.retrieve_entity_labels_in_batch(graph_index, entity_index)]