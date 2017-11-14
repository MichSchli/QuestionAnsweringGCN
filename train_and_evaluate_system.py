import argparse

from evaluation.python_evaluator import Evaluator
from experiment_construction.experiment_factory import ExperimentFactory
from helpers.logger import Logger
from helpers.read_conll_files import ConllReader
from helpers.static import Static
from model_construction.model_builder import ModelBuilder
from model_construction.settings_reader import SettingsReader

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--configuration', type=str, help='The location of the configurations file')
parser.add_argument('--version', type=str, help='Optional version number')
parser.add_argument('--save', type=bool, default=False, help='Determines whether to save the best model on disk.')
args = parser.parse_args()

version_string = ".v" + args.version if args.version else ""

algorithm_name = ".".join(args.configuration.split("/")[-1].split(".")[:-1])
log_file_location = "logs/" + algorithm_name + version_string + ".txt"
save_file_location = "stored_models/" + algorithm_name + version_string + ".ckpt"

Static.logger = Logger(log_file_location, console_verbosity=3, logger_verbosity=2)

settings_reader = SettingsReader()
settings = settings_reader.read(args.configuration)

experiment_builder = ExperimentFactory(settings)
experiment_builder.search()

exit()