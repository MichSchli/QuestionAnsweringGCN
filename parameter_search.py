import argparse

from evaluation.python_evaluator import Evaluator
from helpers.read_conll_files import ConllReader
from model_construction.model_builder import ModelBuilder

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--algorithm', type=str, help='The name of the algorithm to be tested')
parser.add_argument('--train_file', type=str, help='The location of the .conll-file to be used for training')
parser.add_argument('--valid_file', type=str, help='The location of the .conll file to be used for validation')
parser.add_argument('--version', type=str, help='Optional version number')
parser.add_argument('--save', type=bool, default=False, help='Determines whether to save the best model on disk.')
args = parser.parse_args()

version_string = ".v" + args.version if args.version else ""

algorithm_name = ".".join(args.algorithm.split("/")[-1].split(".")[:-1])
log_file_location = "logs/" + algorithm_name + version_string + ".txt"
save_file_location = "stored_models/" + algorithm_name + version_string + ".ckpt"

model_builder = ModelBuilder()
model_builder.build(args.algorithm, version=args.version)

gold_reader = ConllReader(args.valid_file)
evaluator = Evaluator(gold_reader)

best_model = None
best_string = None
best_performance = -1

print("Running parameter tuning for \'" + algorithm_name + "\'.")
if args.version:
    print("Version number: "+args.version)

log_file = open(log_file_location, "w")
for parameter_line, model in model_builder.search():
    print(parameter_line, end="\t")
    print(parameter_line, end="\t", file=log_file)
    train_file_iterator = ConllReader(args.train_file)
    model.train(train_file_iterator)

    valid_file_iterator = ConllReader(args.valid_file)
    prediction = model.predict(valid_file_iterator)
    evaluation = evaluator.evaluate(prediction)
    performance = evaluation.micro_f1

    print(performance)
    print(performance, file=log_file)

    if best_performance < performance:
        best_performance = performance
        best_model = model
        best_string = parameter_line

        if args.save:
            model.save(save_file_location)

print("Parameter tuning done.")
print("Best setting: ")
print(best_string + "\t" + str(best_performance))
