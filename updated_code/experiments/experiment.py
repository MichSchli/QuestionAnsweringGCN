class Experiment:

    example_reader = None

    def __init__(self, example_reader):
        self.example_reader = example_reader

    def run(self):
        for example in self.example_reader.iterate('train', shuffle=True):
            print(example)