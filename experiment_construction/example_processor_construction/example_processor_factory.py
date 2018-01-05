from experiment_construction.example_processor_construction.gold_to_index_example_processor import \
    GoldToIndexExampleProcessor
from experiment_construction.example_processor_construction.graph_split_example_processor import \
    GraphSplitExampleProcessor
from experiment_construction.example_processor_construction.name_to_index_example_processor import \
    NameToIndexExampleProcessor
from experiment_construction.example_processor_construction.propagate_scores_example_processor import \
    PropagateScoresExampleProcessor


class ExampleProcessorFactory:

    def __init__(self):
        pass

    def construct_example_processor(self, settings):
        processor = None

        if "split_graph" in settings["training"]:
            processor = GraphSplitExampleProcessor(processor)

        if "propagate_scores" in settings["training"]:
            processor = PropagateScoresExampleProcessor(processor)

        if "project_name" in settings["training"]:
            return NameToIndexExampleProcessor(processor)
        else:
            return GoldToIndexExampleProcessor(processor)