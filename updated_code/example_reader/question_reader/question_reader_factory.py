from example_reader.question_reader.question_reader import QuestionReader


class QuestionReaderFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration):
        dataset_map = {'train': experiment_configuration['dataset']['train_file'],
                       'valid': experiment_configuration['dataset']['valid_file'],
                       'test': experiment_configuration['dataset']['test_file']}

        return QuestionReader(dataset_map)