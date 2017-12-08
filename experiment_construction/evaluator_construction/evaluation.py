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

    def __init__(self):
        self.total_true_positives = 0
        self.total_false_positives = 0
        self.total_false_negatives = 0

        self.macro_precision = 0
        self.macro_recall = 0
        self.macro_f1 = 0

        self.micro_precision = 0
        self.micro_recall = 0
        self.micro_f1 = 0

    def pretty_print(self, method="micro"):
        if method == "micro":
            print("Final results (micro-averaged):")
            print("===============================")
            print("Precision: " + str(self.micro_precision))
            print("Recall: " + str(self.micro_recall))
            print("F1: " + str(self.micro_f1))
        else:
            print("Final results (macro-averaged):")
            print("===============================")
            print("Precision: " + str(self.macro_precision))
            print("Recall: " + str(self.macro_recall))
            print("F1: " + str(self.macro_f1))