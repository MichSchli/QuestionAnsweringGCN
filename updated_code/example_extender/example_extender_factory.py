from example_extender.remove_gold_from_all_except_best_mention import RemoveGoldFromAllExceptBestMention
from example_extender.vertex_subsampler import VertexSubsampler


class ExampleExtenderFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration, mode):
        extender = RemoveGoldFromAllExceptBestMention()
        if "subsampling" in experiment_configuration["training"] and experiment_configuration["training"]["subsampling"] != "None":
            extender = VertexSubsampler(extender, int(experiment_configuration["training"]["subsampling"]))
        return extender