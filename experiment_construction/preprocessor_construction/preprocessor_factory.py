from experiment_construction.preprocessor_construction.preprocessor import Preprocessor
from input_models.hypergraph.hypergraph_preprocessor import HypergraphPreprocessor
from input_models.mask.mask_preprocessor import LookupMaskPreprocessor
from input_models.padded_map.sentence_to_graph_map_preprocessor import SentenceToGraphMapPreprocessor
from input_models.sentences.sentence_preprocessor import SentencePreprocessor
from input_models.static_embedding.static_entity_embedding_preprocessor import StaticEntityEmbeddingPreprocessor


class PreprocessorFactory:

    def __init__(self):
        pass

    def construct_preprocessor(self, index, settings):
        stack_types = self.get_preprocessor_stack_types(settings)
        preprocessor = Preprocessor()

        for preprocessor_type in stack_types:
            self.add_preprocessor(preprocessor, preprocessor_type, index)

        return preprocessor

    def add_preprocessor(self, preprocessor, preprocessor_type, index):
        if preprocessor_type == "hypergraph":
            preprocessor.hypergraph_batch_preprocessor = HypergraphPreprocessor(index.entity_indexer, index.relation_indexer,
                                                                        "neighborhood", "neighborhood_input_model",
                                                                                preprocessor.stack)
            preprocessor.stack = preprocessor.hypergraph_batch_preprocessor
        elif preprocessor_type == "gold":
            preprocessor.stack = LookupMaskPreprocessor("neighborhood_input_model", "entity_vertex_matrix", "gold_entities",
                                                "gold_mask", preprocessor.stack, clean_dictionary=False, mode="train")
        elif preprocessor_type == "sentence":
            preprocessor.stack = SentencePreprocessor(index.word_indexer, "sentence", "question_sentence_input_model",
                                                      preprocessor.stack)
            preprocessor.stack = SentenceToGraphMapPreprocessor(preprocessor.stack)
        elif preprocessor_type == "static_entity_embeddings":
            preprocessor.stack = StaticEntityEmbeddingPreprocessor(index.entity_indexer, "neighborhood_input_model", preprocessor.stack)

    def get_preprocessor_stack_types(self, settings):
        preprocessor_stack_types = ["hypergraph", "gold", "sentence"]
        if "static_entity_embeddings" in settings["model"] and settings["model"]["static_entity_embeddings"] == "True":
            preprocessor_stack_types += ["static_entity_embeddings"]
        return preprocessor_stack_types