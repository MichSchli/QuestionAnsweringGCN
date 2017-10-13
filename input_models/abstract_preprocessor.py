class AbstractPreprocessor:

    def __init__(self, next_preprocessor):
        self.next_preprocessor = next_preprocessor