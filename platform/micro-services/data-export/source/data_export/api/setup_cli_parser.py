from data_export.api.export_dataset import export_dataset

import config
import logging
import os

logger = logging.getLogger(__name__)


def setup_cli_parser(subparsers):

    parser = subparsers.add_parser('export')

    parser.add_argument("-i", "--input-dataset", default="gs://the-peoples-speech-west-europe/peoples-speech-v0.8/unittest.csv", help="The dataset to export.")
    parser.add_argument("-o", "--output-dataset-path", default="gs://the-peoples-speech-west-europe/peoples-speech-v0.8/unittest.tar.gz", help="The path to save the new dataset.")
    parser.add_argument("-c", "--config-file-path", default="", help="The path to the config file.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print out debug messages.")
    parser.add_argument("-vi", "--verbose-info", default=False, action="store_true", help="Print out info messages.")

    parser.set_defaults(func=dispatch)

def dispatch(args):
    arguments = vars(args)

    config = setup_config(arguments)

    setup_logging(config)

    logger.debug("Full config: " + str(config))

    export_dataset(config["output_dataset_path"], config["input_dataset"], config)

def setup_config(dictionary):
    return config.ConfigurationSet(
        config.config_from_env(prefix="MLCOMMONS"),
        config.config_from_yaml(config_path(dictionary), read_from_file=True),
        config.config_from_dict(dictionary),
    )

def config_path(dictionary):
    if os.path.exists(dictionary["config_file_path"]):
        return dictionary["config_file_path"]

    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, ".mlcommons", "config.yaml")
    if os.path.exists(home_config_path):
        return home_config_path

    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "default.yaml")

def setup_logging(arguments):

    logging_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    if arguments["verbose"]:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    elif arguments["verbose_info"]:
        logging.basicConfig(level=logging.INFO, format=logging_format)
    else:
        logging.basicConfig(level=logging.WARNING, format=logging_format)

    root_logger = logging.getLogger()

    if arguments["verbose"]:
        root_logger.setLevel(logging.DEBUG)
    elif arguments["verbose_info"]:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING)

