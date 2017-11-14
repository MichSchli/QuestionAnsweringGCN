from helpers.read_conll_files import ConllReader


class ExperimentRunner:

    learner = None
    train_file_iterator = None

    def set_train_file(self, train_file_location):
        self.train_file_iterator = ConllReader(train_file_location)

    def train_and_validate(self):
        self.learner.initialize()
        performance = self.learner.train_and_validate(self.train_file_iterator)
        return performance