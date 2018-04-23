import numpy as np


class RemoveGoldFromAllExceptBestMention:

    inner = None

    def __init__(self, inner):
        self.inner = inner

    def extend(self, example):
        example = self.inner.extend(example)

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

        new_gold_list = []

        for i,gold_answer in enumerate(example.gold_answers):
            for j in range(mention_gold_counts.shape[0]):
                if mention_gold_counts[j] == max_count and mention_gold_lists[j][i] == True and gold_answer not in new_gold_list:
                    new_gold_list.append(gold_answer)

        example.gold_answers = new_gold_list

        return example