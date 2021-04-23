
from argparse import ArgumentParser
from data_book.api import import_data_book

import logging

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("Data book API exposed as a command line utility.")

    parser.add_argument("-c", "--max-word-count", default=1e9, help="How many words to add to import to the data book.")
    parser.add_argument("-l", "--language", default="dataset", help="The name of the language to import.")
    parser.add_argument("-i", "--import", default="fasttext", help="Where to import the data book from.")
    parser.add_argument("-o", "--output-path", default="dataset.csv", help="The path to save selected frames.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print out debug messages.")
    parser.add_argument("--verbose-info", default=False, action="store_true", help="Print out info messages.")

    arguments = vars(parser.parse_args())

    setup_logging(arguments)

    import_data_book(arguments)

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

if __name__ == "__main__":
    main()



