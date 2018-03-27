class Batch:

    examples = None

    def __init__(self):
        self.examples = []

    def count_examples(self):
        return len(self.examples)

    def has_examples(self):
        return len(self.examples) > 0

    def get_examples(self):
        return self.examples