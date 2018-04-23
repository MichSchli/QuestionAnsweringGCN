class TensorflowVariablesHolder:

    variables = None
    assignments = None

    def __init__(self):
        self.variables = {}
        self.assignments = {}

    def add_variable(self, variable_name, variable):
        if variable_name in self.variables.keys():
            print("Warning: Variable name reused -- "+variable_name)

        self.variables[variable_name] = variable

    def get_variable(self, variable_name):
        return self.variables[variable_name]

    def assign_variable(self, name, value):
        variable = self.variables[name]
        self.assignments[variable] = value

    def get_assignment_dict(self):
        return self.assignments
