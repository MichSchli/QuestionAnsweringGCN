from example_reader.mention_reader.mention import Mention
import numpy as np


class MentionReader:

    entity_prefix_in_db = None

    def __init__(self, entity_prefix_in_db):
        self.entity_prefix_in_db

    def build(self, array_mentions):
        mentions = []
        for mention_line in array_mentions:
            mention = Mention()
            mention.word_indexes = np.arange(int(mention_line[0]), int(mention_line[1])+1, dtype=np.int32)
            mention.entity_label = self.entity_prefix_in_db + mention_line[2]
            mention.score = mention_line[3]
            mentions.append(mention)

        return mentions