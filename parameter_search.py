import argparse

from evaluation.python_evaluator import Evaluator
from experiment_construction.model_construction.validation_set_evaluator import ValidationSetEvaluator
from helpers.logger import Logger
from helpers.read_conll_files import ConllReader
from helpers.static import Static
from model_construction.model_builder import ModelBuilder
from model_construction.settings_reader import SettingsReader

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--algorithm', type=str, help='The location of the model config file')
parser.add_argument('--dataset', type=str, help='The location of the dataset config file')
parser.add_argument('--version', type=str, help='Optional version number')
parser.add_argument('--save', type=bool, default=False, help='Determines whether to save the best model on disk.')
args = parser.parse_args()

version_string = ".v" + args.version if args.version else ""

algorithm_name = ".".join(args.algorithm.split("/")[-1].split(".")[:-1])
log_file_location = "logs/" + algorithm_name + version_string + ".txt"
save_file_location = "stored_models/" + algorithm_name + version_string + ".ckpt"

Static.logger = Logger(log_file_location, console_verbosity=3, logger_verbosity=2)

settings = {}
settings_reader = SettingsReader()
settings["algorithm"] = settings_reader.read(args.algorithm)
settings["dataset"] = settings_reader.read(args.dataset)

model_builder = ModelBuilder()
model_builder.build(settings, version=args.version)

gold_reader = ConllReader(settings["dataset"]["location"]["valid_file"])
evaluator = Evaluator(gold_reader)

best_model = None
best_string = None
best_performance = -1

Static.logger.write("Running parameter tuning for \'" + algorithm_name + "\'.", verbosity_priority=1)
if args.version:
    Static.logger.write("Version number: "+args.version, verbosity_priority=1)

for parameter_line, model in model_builder.search():
    Static.logger.write(parameter_line, verbosity_priority=2)
    train_file_iterator = ConllReader(settings["dataset"]["location"]["train_file"])
    model = ValidationSetEvaluator(model, settings)
    epoch, performance = model.train(train_file_iterator)

    if best_performance < performance:
        best_performance = performance
        best_model = model
        best_string = parameter_line

        if args.save:
            model.save(save_file_location)

Static.logger.write("Parameter tuning done.", verbosity_priority=1)
Static.logger.write("Best setting: ", verbosity_priority=1)
Static.logger.write(best_string + "\t" + str(best_performance), verbosity_priority=1)
