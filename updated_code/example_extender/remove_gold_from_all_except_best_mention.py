import numpy as np


class RemoveGoldFromAllExceptBestMention:

    def __init__(self):
        pass

    def extend(self, example, mode):
        if mode != "train":
            return example

        mention_gold_lists = np.zeros((len(example.mentions), len(example.gold_answers)), dtype=np.bool)

        for i,gold_answer in enumerate(example.gold_answers):
            entity_indexes = gold_answer.entity_indexes
            for index in entity_indexes:
                nearby_mentions = example.graph.get_nearby_centroids(index)
                for j, mention in enumerate(example.mentions):
                    if mention.entity_index in nearby_mentions:
                        mention_gold_lists[j][i] = True

        mention_gold_counts = np.sum(mention_gold_lists, axis=1)
        max_count = np.max(mention_gold_counts)

        new_mention_list = []

        for i in range(len(mention_gold_counts)):
            if mention_gold_counts[i] == max_count:
                new_mention_list.append(example.mentions[i])

        example.mentions = new_mention_list
        return example