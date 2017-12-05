class AbstractComponent:

    def get_regularization_term(self):
        return 0


    def prepare_tensorflow_variables(self, mode="train"):
        pass

    def handle_variable_assignment(self, batch, mode):
        pass