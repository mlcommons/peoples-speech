
from smart_open import open

import csv

import json

import logging

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

def load_dataset_csv(csv_path, task_id, task_count):

    with open(csv_path) as csv_file:
        return (yield from load_dataset_csv_from_file(csv_file, task_id, task_count))

def load_dataset_csv_from_file(csv_file, task_id, task_count):
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        index = 0
        for row in reader:
            path, caption = row[0], row[1]

            metadata = {}
            if len(row) >= 3:
                metadata = json.loads(row[2])

            if index % task_count == task_id:
                yield {"path" : path, "caption" : caption, "metadata" : metadata}

            index += 1
