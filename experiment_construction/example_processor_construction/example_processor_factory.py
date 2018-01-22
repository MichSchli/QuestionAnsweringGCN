from experiment_construction.example_processor_construction.gold_to_index_example_processor import \
    GoldToIndexExampleProcessor
from experiment_construction.example_processor_construction.graph_split_example_processor import \
    GraphSplitExampleProcessor
from experiment_construction.example_processor_construction.name_to_index_example_processor import \
    NameToIndexExampleProcessor
from experiment_construction.example_processor_construction.propagate_scores_example_processor import \
    PropagateScoresExampleProcessor
from experiment_construction.example_processor_construction.simple_split_example_processor import \
    SimpleSplitExampleProcessor
from experiment_construction.example_processor_construction.subsample_vertices_example_processor import \
    SubsampleVerticesExampleProcessor


class ExampleProcessorFactory:

    def __init__(self):
        pass

    def construct_example_processor(self, settings):
        processor = None

        if "split_graph" in settings["training"] and settings["training"]["split_graph"] == "True":
            processor = GraphSplitExampleProcessor(processor)

        processor = SimpleSplitExampleProcessor(processor)

        if "project_name" in settings["training"]:
            processor = NameToIndexExampleProcessor(processor)
        else:
            processor = GoldToIndexExampleProcessor(processor)

        if "subsampling" in settings["training"] and settings["training"]["subsampling"] != "None":
            processor = SubsampleVerticesExampleProcessor(processor, rate=int(settings["training"]["subsampling"]))

        if "propagate_scores" in settings["training"] and settings["training"]["propagate_scores"] == "True":
            processor = PropagateScoresExampleProcessor(processor)

        return processor
