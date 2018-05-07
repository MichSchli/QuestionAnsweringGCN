class Gcn():

    propagators = None
    updaters = None

    def __init__(self, initializer, propagators, updaters):
        self.initializer = initializer
        self.propagators = propagators
        self.updaters = updaters

        self.variables = {}

    def run(self, mode):
        carry_over = self.initializer.initialize(mode)

        for layer, updater in zip(self.propagators, self.updaters):
            incoming_messages = 0
            for propagator in layer:
                incoming_messages += propagator.propagate(mode)

            carry_over = updater.update(incoming_messages, carry_over)

    def get_regularization(self):
        reg = 0

        for layer in self.propagators:
            for propagator in layer:
                reg += propagator.get_regularization()

        for updater in self.updaters:
            reg += updater.get_regularization()

        return reg

    def handle_variable_assignment(self, batch, mode):
        pass