import argparse

from experiment_construction.experiment_factory import ExperimentFactory
from helpers.logger import Logger
from helpers.static import Static
from experiment_construction.settings_reader import SettingsReader

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--configuration', type=str, help='The location of the configurations file')
parser.add_argument('--version', type=str, help='Optional version number')
parser.add_argument('--save', type=bool, default=False, help='Determines whether to save the best model on disk.')
parser.add_argument('--test', action="store_true", help='Test the best model on all three datasets after searching.')
parser.add_argument('--logger_configuration', help='Configuration file for the logger.')
args = parser.parse_args()

version_string = ".v" + args.version if args.version else ""

algorithm_name = ".".join(args.configuration.split("/")[-1].split(".")[:-1])
log_file_location = "logs/" + algorithm_name + version_string + ".txt"
save_file_location = "stored_models/" + algorithm_name + version_string + ".ckpt"

settings_reader = SettingsReader()
settings = settings_reader.read(args.configuration)
logger_settings = settings_reader.read(args.logger_configuration)

Static.logger = Logger(log_file_location, logger_settings)

experiment_builder = ExperimentFactory(settings)
best_configuration = experiment_builder.search()

if args.test:
    experiment_builder.train_and_validate(best_configuration)
    experiment_builder.evaluate("train_file")
    experiment_builder.evaluate("valid_file")
    experiment_builder.evaluate("test_file")