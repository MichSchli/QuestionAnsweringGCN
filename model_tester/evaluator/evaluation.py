class Evaluation:

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0

    micro_precision = 0
    micro_recall = 0
    micro_f1 = 0

    n_samples = 0

    def __init__(self, default_method):
        self.total_true_positives = 0
        self.total_false_positives = 0
        self.total_false_negatives = 0

        self.macro_precision = 0
        self.macro_recall = 0
        self.macro_f1 = 0

        self.micro_precision = 0
        self.micro_recall = 0
        self.micro_f1 = 0

        self.n_samples = 0

        self.default_method=default_method

    def get_average_positives(self):
        return (self.total_false_positives + self.total_true_positives)/self.n_samples

    def temporary_summary(self):
        string = "Final results (macro-averaged):\n"
        string += "===============================\n"
        string += "Precision: " + str(self.macro_precision/self.n_samples) + "\n"
        string += "Recall: " + str(self.macro_recall/self.n_samples) + "\n"
        string += "F1: " + str(self.macro_f1/self.n_samples) + "\n"
        string += "Average positives: " + str(self.get_average_positives()) + "\n"
        string += "Count: " + str(self.n_samples) + "\n"

        return string

    def summary(self, method=None):
        if method is None:
            method = self.default_method

        string = ""
        if method == "micro":
            string += "Final results (micro-averaged):\n"
            string += "===============================\n"
            string += "Precision: " + str(self.micro_precision) + "\n"
            string += "Recall: " + str(self.micro_recall) + "\n"
            string += "F1: " + str(self.micro_f1) + "\n"
            string += "Average positives: " + str(self.get_average_positives()) + "\n"
        else:
            string += "Final results (macro-averaged):\n"
            string += "===============================\n"
            string += "Precision: " + str(self.macro_precision) + "\n"
            string += "Recall: " + str(self.macro_recall) + "\n"
            string += "F1: " + str(self.macro_f1) + "\n"
            string += "Average positives: " + str(self.get_average_positives()) + "\n"
        return string