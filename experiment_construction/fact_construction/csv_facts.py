class CsvFacts:

    number_of_triples = None
    number_of_relation_types = None
    number_of_entities = None

    def __init__(self, filename, separator=","):
        self.number_of_triples = 0
        self.number_of_relation_types = 0
        self.number_of_entities = 0

        seen_relation_types = set([])
        seen_entities = set([])

        for line in open(filename):
            self.number_of_triples += 1
            parts = line.strip().split(separator)

            if parts[0] not in seen_entities:
                seen_entities.add(parts[0])
                self.number_of_entities += 1

            if parts[2] not in seen_entities:
                seen_entities.add(parts[2])
                self.number_of_entities += 1

            if parts[1] not in seen_relation_types:
                seen_entities.add(parts[0])
                self.number_of_relation_types += 1