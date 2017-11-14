class GreedySearch:

    default_configuration = None
    option_iterator = None
    best_performances = None
    best_configurations = None
    current_option_key = None

    def __init__(self, settings):
        self.default_configuration = self.get_default_configuration(settings)
        self.option_iterator = self.get_option_iterator(settings)
        self.best_performances = {}
        self.best_configurations = {}
        self.current_option_key = None

    def next(self, previous_score):
        option = self.option_iterator.__next__()
        if option is None:
            self.update_best_performance(previous_score)

            if self.current_option_key is not None:
                self.default_configuration[self.current_option_key[0]][self.current_option_key[1]] = self.best_configurations[self.current_option_key]
            return None

        option_key = (option[0], option[1])
        if option_key not in self.best_performances:
            self.initialize_best_performance(option, option_key)

        self.update_best_performance(previous_score)

        if self.current_option_key != option_key:
            if self.current_option_key is not None:
                self.default_configuration[self.current_option_key[0]][self.current_option_key[1]] = self.best_configurations[self.current_option_key]
            self.current_option_key = option_key

        self.default_configuration[option[0]][option[1]] = option[2]

        return self.default_configuration

    def initialize_best_performance(self, option, option_key):
        if self.current_option_key is not None:
            self.best_performances[option_key] = self.best_performances[self.current_option_key]
            self.best_configurations[option_key] = self.default_configuration[option[0]][option[1]]
        else:
            self.best_performances[option_key] = None

    def update_best_performance(self, previous_score):
        if self.current_option_key is not None \
                and (self.best_performances[self.current_option_key] is None \
                             or previous_score > self.best_performances[self.current_option_key]):
            self.best_configurations[self.current_option_key] = \
                self.default_configuration[self.current_option_key[0]][self.current_option_key[1]]
            self.best_performances[self.current_option_key] = previous_score

    def get_default_configuration(self, settings):
        d = {}
        for header, options in settings.items():
            d[header] = {}
            for k, v in options.items():
                d[header][k] = v.split(",")[0]
        return d

    def get_option_iterator(self, settings):
        first = True
        all_options = []

        stored_default = None

        for header, options in settings.items():
            for k, v in options.items():
                options = [(header, k, option) for option in v.split(",")]
                if first and len(options) > 1:
                    first = False
                    all_options.append(options[0])

                all_options.extend(options[1:])
                stored_default = options[0]

        for option in all_options:
            yield option

        if len(all_options) == 0:
            yield stored_default

        yield None
