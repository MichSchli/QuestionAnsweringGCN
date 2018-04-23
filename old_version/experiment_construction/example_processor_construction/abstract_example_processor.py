class AbstractExampleProcessor:

    next = None

    def __init__(self, next):
        self.next = next

    def process_stream(self, example_stream, mode="train"):
        if self.next is not None:
            example_stream = self.next.process_stream(example_stream, mode=mode)

        for example in example_stream:
            if self.process_example(example, mode=mode):
                yield example