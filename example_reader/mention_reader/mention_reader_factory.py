from example_reader.mention_reader.mention_reader import MentionReader


class MentionReaderFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration):
        if "prefix" in experiment_configuration["endpoint"]:
            prefix = experiment_configuration["endpoint"]["prefix"]
        else:
            prefix = ""

        mention_reader = MentionReader(prefix)

        if "transform_mention_scores" in experiment_configuration["dataset"] \
                and experiment_configuration["dataset"]["transform_mention_scores"] == "log":
            mention_reader.set_score_transform("log")

        return mention_reader