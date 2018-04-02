import numpy as np


class Example:

    question = None
    mentions = None
    gold_answers = None
    graph = None
    mentions = None
    prediction = None

    def __init__(self):
        self.mentions = []

    def __str__(self):
        return "Example:\""+" ".join([word for word in self.question.words]) + "\"" \
               + "\n\t" + "\n\t".join([str(g) for g in self.gold_answers])

    def get_mentioned_entities(self):
        unique_entities = []
        for mention in self.mentions:
            if mention.entity_label not in unique_entities:
                unique_entities.append(mention.entity_label)

        return unique_entities

    def index_mentions(self):
        for mention in self.mentions:
            mention.entity_index = self.graph.map_from_label(mention.entity_label)

    def index_gold_answers(self):
        for gold_answer in self.gold_answers:
            if not self.project_names:
                gold_answer.entity_indexes = np.array([self.graph.map_from_label(gold_answer.entity_name_or_label)])
            else:
                gold_answer.entity_indexes = np.array(self.graph.map_from_name_or_label(gold_answer.entity_name_or_label))

    def get_centroid_indexes(self):
        centroid_indexes = []
        for mention in self.mentions:
            centroid_indexes.append(mention.entity_index)
        return np.unique(centroid_indexes)

    def get_gold_indexes(self):
        gold_indexes = []
        for gold_answer in self.gold_answers:
            gold_indexes.extend(gold_answer.entity_indexes)
        return np.unique(gold_indexes)

    def get_gold_labels(self):
        gold_labels = []
        for gold_answer in self.gold_answers:
            gold_labels.append(gold_answer.entity_name_or_label)
        return np.unique(gold_labels)

    def get_predicted_labels(self, threshold):
        return self.prediction.get_predictions(threshold)

    def count_entities(self):
        return self.graph.entity_vertex_indexes.shape[0]

    def get_entity_vertex_indexes(self):
        return self.graph.entity_vertex_indexes

    def count_vertices(self):
        return self.graph.count_vertices()

    def count_words(self):
        return self.question.count_words()

    def get_padded_edge_part_type_matrix(self):
        return self.graph.padded_edge_bow_matrix

    def count_edges(self):
        return self.graph.edges.shape[0]

    def has_mentions(self):
        return len(self.mentions) > 0