
from bs4 import BeautifulSoup

from warcio.archiveiterator import ArchiveIterator

import csv
import cld3

from argparse import ArgumentParser

import logging

logger = logging.getLogger(__name__)

def main():

    parser = ArgumentParser(description="Convert WARC archive to csv, one line per entry.")


    parser.add_argument("-o", "--output-path", default="",
        help = "The input path to read a warc from.")

    parser.add_argument("-c", "--count", default=1e4,
        help = "How many websites to extract.")
    parser.add_argument("-l", "--language", default="en",
        help = "The language of the text to extract.")

    parser.add_argument("-i", "--input-path", default="",
        help = "The output path the save csv to.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    convert_warc_to_csv(arguments)

def convert_warc_to_csv(arguments):
    counter = 0
    with open(arguments["input_path"], 'rb') as input_file, \
         open(arguments["output_path"], "w", newline='') as output_file:

        writer = csv.writer(output_file, delimiter=',', quotechar='"')

        for record in ArchiveIterator(input_file):

            if record.rec_type == 'response':
                if record.http_headers.get_header('Content-Type') == 'text/html':
                    html = record.content_stream().read()
                    clean_text = clean_html(html)

                    if len(clean_text) > 0:
                        language_prediction = cld3.get_language(clean_text)
                        if language_prediction.language == arguments["language"]:
                            writer.writerow([clean_text, language_prediction])
                            counter += 1

                            if counter >= int(arguments["count"]):
                                return

                            if counter % 100 == 0:
                                logger.info("Saved " + str(counter) + " websites")



def clean_html(html):
    soup = BeautifulSoup(html, features="html.parser", )

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    text = text.replace('\x00','')

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    return "\n".join([chunk for chunk in chunks if chunk])

def setup_logger(arguments):

   if arguments["verbose"]:
       logger.setLevel(logging.DEBUG)
   else:
       logger.setLevel(logging.INFO)

   ch = logging.StreamHandler()
   ch.setLevel(logging.DEBUG)

   # create formatter
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

   # add formatter to ch
   ch.setFormatter(formatter)

   # add ch to logger
   logger.addHandler(ch)


if __name__ == "__main__":
    main()



