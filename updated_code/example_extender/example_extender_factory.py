from example_extender.add_inverse_edge_extender import AddInverseEdgeExtender
from example_extender.empty_extender import EmptyExtender
from example_extender.remove_gold_from_all_except_best_mention import RemoveGoldFromAllExceptBestMention
from example_extender.vertex_subsampler import VertexSubsampler


class ExampleExtenderFactory:

    index_factory = None

    def __init__(self, index_factory):
        self.index_factory = index_factory

    def get(self, experiment_configuration, mode):
        if mode == "train":
            extender = EmptyExtender()
            if "filter_gold_labels" in experiment_configuration["training"] and experiment_configuration["training"]["filter_gold_labels"] == "maximize_f1":
                extender = RemoveGoldFromAllExceptBestMention(extender)

            if "subsampling" in experiment_configuration["training"] and experiment_configuration["training"]["subsampling"] != "None":
                extender = VertexSubsampler(extender, int(experiment_configuration["training"]["subsampling"]))
        else:
            extender = EmptyExtender()

        relation_index = self.index_factory.get("relations", experiment_configuration)
        extender = AddInverseEdgeExtender(extender, relation_index)

        return extender