import argparse

from evaluation.python_evaluator import Evaluator
from helpers.read_conll_files import ConllReader
from model_construction.model_builder import ModelBuilder

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--algorithm', type=str, help='The name of the algorithm to be tested')
parser.add_argument('--train_file', type=str, help='The location of the .conll-file to be used for training')
parser.add_argument('--valid_file', type=str, help='The location of the .conll file to be used for validation')
parser.add_argument('--version', type=str, help='Optional version number')
args = parser.parse_args()

model_builder = ModelBuilder()
model_builder.build(args.algorithm, version=args.version)

gold_reader = ConllReader(args.valid_file)
evaluator = Evaluator(gold_reader)

best_model = None
best_performance = -1

for parameter_line, model in model_builder.search():
    print(parameter_line, end="\t")
    train_file_iterator = ConllReader(args.train_file)
    model.train(train_file_iterator)

    evaluation = evaluator.evaluate(model)
    performance = evaluation.f1

    print(performance)

    if best_performance < performance:
        best_performance = performance
        best_model = model

best_model.save()
