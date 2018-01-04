"""
Different candidate selectors
"""
from candidate_selection.tensorflow_models.baselines.entity_embedding_vs_bag_of_words import EntityEmbeddingVsBagOfWords
from candidate_selection.tensorflow_models.baselines.entity_embedding_vs_gold import EntityEmbeddingVsGold
from candidate_selection.tensorflow_models.baselines.entity_embedding_vs_lstm import EntityEmbeddingVsLstm
from candidate_selection.tensorflow_models.baselines.path_bag_vs_bag_of_words import PathBagVsBagOfWords
from candidate_selection.tensorflow_models.baselines.path_bag_vs_lstm import PathBagVsLstm
from candidate_selection.tensorflow_models.components.extras.mean_gold_embedding_retriever import \
    MeanGoldEmbeddingRetriever
from candidate_selection.test_models.oracle_candidate import OracleCandidate


class CandidateSelectorFactory:

    def __init__(self):
        pass

    def construct_candidate_selector(self, indexers, facts, preprocessors, settings):
        model_class = self.retrieve_class_name(settings["model"]["stack_name"])
        model = model_class(facts)

        for k, v in settings["model"].items():
            if k == "stack_name":
                continue
            else:
                model.update_setting(k, v)

        model.set_indexers(indexers)
        model.set_preprocessor(preprocessors)

        return model

    def retrieve_class_name(self, stack_name):
        if stack_name == "oracle":
            return OracleCandidate

        if stack_name == "bow+dumb":
            return EntityEmbeddingVsBagOfWords

        if stack_name == "lstm+dumb":
            return EntityEmbeddingVsLstm

        if stack_name == "lstm+path":
            return PathBagVsLstm

        if stack_name == "bow+path":
            return PathBagVsBagOfWords

        if stack_name == "gold_retriever":
            return EntityEmbeddingVsGold

        return None