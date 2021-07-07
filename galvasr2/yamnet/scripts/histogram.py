import os

from argparse import ArgumentParser

import config

import logging

from smart_open import open
import jsonlines

import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def main():
    config = parse_arguments()

    setup_logging(config)

    make_histogram(config)

def make_histogram(config):
    filtered = set(["path", "seconds"])
    histogram = {}
    with open(config["input_path"]) as jsonlfile:
        with jsonlines.Reader(jsonlfile) as reader:
            for item in reader:
                filtered_keys = []

                for key, value in item.items():
                    if key in filtered:
                        continue

                    filtered_keys.append((value, key))

                keys = [(value, key) for value, key in filtered_keys if value > 0.0]

                for weight, key in keys:
                    if not key in histogram:
                        histogram[key] = 0.0

                    histogram[key] += weight

    print(histogram)
    plot_histogram(histogram)

def plot_histogram(histogram):
    sns.set()
    sns.set_palette("bright")
    sns.set(font_scale=1.6)
    counts = dict(sorted(histogram.items(), key=lambda item: item[1], reverse=True)[2:22])
    fig, ax = plt.subplots()
    ax.bar(counts.keys(), [(113* value * 15. / 3600) for value in counts.values()])
    ax.set_xticklabels(counts.keys(), rotation=70)
    #ax.set_ylim([0, 150])
    fig.set_size_inches(20, 8)
    plt.show()

def parse_arguments():
    parser = ArgumentParser("Generate a histogram from yamnet classes.")

    parser.add_argument("-i", "--input-path",
        default="gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/yamnet.jsonl",
        help="Path to yamnet results.")
    parser.add_argument("-o", "--output-path", default="results.jsonl", help="The path to save the results.")
    parser.add_argument("-c", "--config-file-path", default=".json", help="The path to the config file.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print out debug messages.")
    parser.add_argument("-vi", "--verbose-info", default=False, action="store_true", help="Print out info messages.")

    args = parser.parse_args()
    arguments = vars(args)

    config = setup_config(arguments)

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

    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "default.yaml")

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

    logging.getLogger("numba.core.ssa").setLevel(logging.CRITICAL)
    logging.getLogger("numba.core.byteflow").setLevel(logging.CRITICAL)
    logging.getLogger("numba.core.interpreter").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
    logging.getLogger("google.resumable_media").setLevel(logging.CRITICAL)

if __name__ == '__main__':
    main()
