from example_extender.add_golds_with_similar_path_bags import AddGoldsWithSimilarPathBags
from example_extender.add_inverse_edge_extender import AddInverseEdgeExtender
from example_extender.add_max_score_extender import AddMaxScoreExtender
from example_extender.empty_extender import EmptyExtender
from example_extender.remove_gold_from_all_except_best_mention import RemoveGoldFromAllExceptBestMention
from example_extender.vertex_subsampler import VertexSubsampler

from example_extender.add_dependency_edges import AddDependencyEdgeExtender
from example_extender.add_mention_dummy_extender import AddMentionDummyExtender
from example_extender.add_word_sequence_edges import AddWordSequenceEdgeExtender
from example_extender.add_word_vertices_extender import AddWordDummyExtender


class ExampleExtenderFactory:

    index_factory = None

    def __init__(self, index_factory):
        self.index_factory = index_factory

    def get(self, experiment_configuration, mode):
        if mode == "train":
            extender = EmptyExtender()

            extender = AddGoldsWithSimilarPathBags(extender, 0.8, project_names=experiment_configuration["endpoint"]["project_names"] == "True")

            if "filter_gold_labels" in experiment_configuration["training"] and experiment_configuration["training"]["filter_gold_labels"] == "maximize_f1":
                extender = RemoveGoldFromAllExceptBestMention(extender)

            if "subsampling" in experiment_configuration["training"] and experiment_configuration["training"]["subsampling"] != "None":
                extender = VertexSubsampler(extender, int(experiment_configuration["training"]["subsampling"]))
        else:
            extender = EmptyExtender()

        relation_index = self.index_factory.get("relations", experiment_configuration)
        entity_index = self.index_factory.get("vertices", experiment_configuration)

        extender = AddMaxScoreExtender(extender)

        extender = AddMentionDummyExtender(extender, relation_index, entity_index)
        extender = AddWordDummyExtender(extender, relation_index, entity_index)
        extender = AddWordSequenceEdgeExtender(extender, relation_index)
        extender = AddDependencyEdgeExtender(extender, relation_index, entity_index)

        if "inverse_relations" in experiment_configuration["architecture"] and experiment_configuration["architecture"]["inverse_relations"] == "features":
            extender = AddInverseEdgeExtender(extender, relation_index)

        return extender