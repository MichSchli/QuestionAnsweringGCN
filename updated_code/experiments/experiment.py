class Experiment:

    example_reader = None

    def __init__(self, example_reader, example_extender):
        self.example_reader = example_reader
        self.example_extender = example_extender

    def run(self):
        for example in self.example_reader.iterate('train', shuffle=True):
            example = self.example_extender.extend(example, 'train')
            print(example)