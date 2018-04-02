from example_reader.mention_reader.mention_reader import MentionReader


class MentionReaderFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration):
        if "prefix" in experiment_configuration["endpoint"]:
            prefix = experiment_configuration["endpoint"]["prefix"]
        else:
            prefix = ""

        return MentionReader(prefix)