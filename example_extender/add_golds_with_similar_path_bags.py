import numpy as np

from example_reader.gold_answer_reader.gold_answer import GoldAnswer


class AddGoldsWithSimilarPathBags:

    inner = None
    similarity = None
    project_names = None

    def __init__(self, inner, similarity, project_names):
        self.inner = inner
        self.similarity = similarity
        self.project_names = project_names

    def extend(self, example):
        example = self.inner.extend(example)
        has_printed = False

        additional_gold_answers = []

        for j in example.graph.entity_vertex_indexes:
            if j in example.get_gold_indexes():
                continue

            add = False

            for i, gold_answer in enumerate(example.gold_answers):
                for index in gold_answer.entity_indexes:
                    gold_path_bag = np.array(example.graph.entity_centroid_paths[index])
                    j_path_bag = np.array(example.graph.entity_centroid_paths[j])
                    overlap = np.in1d(gold_path_bag, j_path_bag)
                    overlap_rate = np.sum(overlap) / len(gold_path_bag)

                    if overlap_rate > self.similarity:
                        if not has_printed:
                            print(example)
                            has_printed = True

                        print(overlap_rate)
                        # NOTE: MAybe score according to overlap
                        add = True

            if add:
                gold_answer = GoldAnswer()
                if self.project_names:
                    gold_answer.entity_name_or_label = example.graph.map_to_name_or_label(j)
                    gold_answer.entity_indexes = np.array(
                        example.graph.map_from_name_or_label(gold_answer.entity_name_or_label))
                else:
                    gold_answer.entity_name_or_label = example.graph.map_to_label(j)
                    gold_answer.entity_indexes = np.array(
                        [example.graph.map_from_label(gold_answer.entity_name_or_label)])

                gold_answer.entity_indexes = gold_answer.entity_indexes[
                    np.where(gold_answer.entity_indexes >= 0)]

                additional_gold_answers.append(gold_answer)
                example.gold_answers.append(gold_answer)

        if len(additional_gold_answers) > 0:
            print(example)
            print("====================================")

        return example