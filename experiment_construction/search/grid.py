import itertools


class GridSearch:

    iterator = None

    def __init__(self, settings):
        self.iterator = self.grid_search(settings)

    def next(self, previous_score):
        return self.iterator.__next__()

    """
    Iterate settings using grid search:
    """

    def grid_search(self, settings):
        configurations = []

        for header, options in settings.items():
            for k, v in options.items():
                options = [(header, k, option) for option in v.split(",")]
                configurations.append(options)

        for combination in itertools.product(*configurations):
            combination_dictionary = {}
            for part in combination:
                if part[0] not in combination_dictionary:
                    combination_dictionary[part[0]] = {}

                combination_dictionary[part[0]][part[1]] = part[2]

            yield combination_dictionary
        yield None