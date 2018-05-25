from example_reader.gold_path_reader.gold_path_finder import GoldPathFinder


class GoldPathReaderFactory:

    def __init__(self, index_factory):
        self.index_factory = index_factory

    def get(self, experiment_configuration):
        dataset_map = {'train': experiment_configuration['dataset']['train_gold_relations'],
                       'valid': experiment_configuration['dataset']['valid_gold_relations'],
                       'test': experiment_configuration['dataset']['test_gold_relations']}

        relation_index = self.index_factory.get("relations", experiment_configuration)

        return GoldPathFinder(relation_index, dataset_map)