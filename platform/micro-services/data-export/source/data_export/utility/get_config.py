
import config
import os
import logging

def get_config():
    config = setup_config({})

    setup_logging(config)

    return config

def setup_config(dictionary):
    return config.ConfigurationSet(
        config.config_from_env(prefix="MLCOMMONS"),
        config.config_from_yaml(config_path(), read_from_file=True),
        config.config_from_dict(dictionary),
    )

def config_path():
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

