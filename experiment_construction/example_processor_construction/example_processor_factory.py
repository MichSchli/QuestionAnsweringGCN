from experiment_construction.example_processor_construction.gold_by_f1_filter_example_processor import \
    GoldByF1FilterExampleProcessor
from experiment_construction.example_processor_construction.gold_to_index_example_processor import \
    GoldToIndexExampleProcessor
from experiment_construction.example_processor_construction.graph_split_example_processor import \
    GraphSplitExampleProcessor
from experiment_construction.example_processor_construction.name_to_index_example_processor import \
    NameToIndexExampleProcessor
from experiment_construction.example_processor_construction.propagate_scores_example_processor import \
    PropagateScoresExampleProcessor
from experiment_construction.example_processor_construction.subsample_vertices_example_processor import \
    SubsampleVerticesExampleProcessor


class ExampleProcessorFactory:

    def __init__(self):
        pass

    def construct_example_processor(self, settings):
        processor = self.add_gold_label_projectors(None, settings)
        processor = self.add_graph_editing_processors(processor, settings)
        processor = self.add_gold_label_filters(processor, settings)

        if "propagate_scores" in settings["training"] and settings["training"]["propagate_scores"] == "True":
            processor = PropagateScoresExampleProcessor(processor)

        return processor

    '''
    Edit the graph to fit the model:
    '''
    def add_graph_editing_processors(self, processor, settings):
        if "split_graph" in settings["training"] and settings["training"]["split_graph"] == "True":
            processor = GraphSplitExampleProcessor(processor)
        if "subsampling" in settings["training"] and settings["training"]["subsampling"] != "None":
            processor = SubsampleVerticesExampleProcessor(processor, rate=int(settings["training"]["subsampling"]))
        return processor

    '''
    Process gold labels:
    '''
    def add_gold_label_projectors(self, processor, settings):
        if "project_name" in settings["training"] and settings["training"]["project_name"] == "True":
            processor = NameToIndexExampleProcessor(processor)
        else:
            processor = GoldToIndexExampleProcessor(processor)

        return processor

    '''
    Filter gold labels:
    '''
    def add_gold_label_filters(self, processor, settings):
        if settings["training"]["gold_labels"] == "maximize_f1":
            processor = GoldByF1FilterExampleProcessor(processor)
        if settings["training"]["gold_labels"] == "span_maximize_f1":
            exit()

        return processor
