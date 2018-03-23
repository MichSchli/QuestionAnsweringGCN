class Example:

    words = None
    mentions = None
    gold_answers = None
    graph = None

    def __str__(self):
        return "Example: \""+" ".join([w[1] for w in self.words]) + "\""

    def get_mentioned_entities(self):
        unique_entities = []
        for mention in self.mentions:
            if mention[2] not in unique_entities:
                unique_entities.append(mention[2])

        return unique_entities