from auxilliaries.settings_reader import SettingsReader
from experiments.experiment_factory import ExperimentFactory
import argparse

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--configuration', type=str, help='The location of the configurations file')
parser.add_argument('--version', type=str, help='Optional version number or description')
parser.add_argument('--logger_configuration', help='Configuration file for the logger.')
args = parser.parse_args()

settings_reader = SettingsReader()
experiment_settings = settings_reader.read(args.configuration)
logger_settings = settings_reader.read(args.logger_configuration)

experiment_factory = ExperimentFactory()
experiment = experiment_factory.get(experiment_settings)

experiment.run()