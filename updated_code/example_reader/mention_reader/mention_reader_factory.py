from example_reader.mention_reader.mention_reader import MentionReader


class MentionReaderFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration):
        return MentionReader(experiment_configuration["endpoint"]["prefix"])