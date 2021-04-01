
import hashlib
import csv
import tempfile
import json
import os
import whoosh.index
import whoosh.fields
import whoosh.qparser

import logging

logger = logging.getLogger(__name__)

class ExistingDatasetSearchEngine:
    def __init__(self, config):
        self.config = config

    def search_for_data_book(self, data_book):
        # load dataset index
        self.load_index()

        # load samples
        results = []
        result_set = set()

        for page in data_book:
            results_for_this_page = self.search_for_page(page)
            results_for_this_page = dedup(results_for_this_page, result_set)
            results.extend(results_for_this_page)

        return results

    def search_for_page(self, data_book_page):

        logger.debug("Searching for page: " + str(data_book_page))

        # figure out how many results per databook page
        results_per_data_book_page = int(self.config["search"]["samples_per_page"])

        label = data_book_page[0]

        with self.index.searcher() as searcher:
            query = whoosh.qparser.QueryParser("label", self.index.schema).parse(label)
            results = searcher.search(query, limit=results_per_data_book_page)
            results = [(hit["label"], hit["path"]) for hit in results]
            logger.debug("Got search results: " + str(results))

        return results

    def load_index(self):
        self.index_path = tempfile.TemporaryDirectory()

        # Create the index
        schema = whoosh.fields.Schema(
            label=whoosh.fields.TEXT(stored=True),
            path=whoosh.fields.ID(stored=True),
            content=whoosh.fields.TEXT)
        self.index = whoosh.index.create_in(self.index_path.name, schema)

        # Load datasets into it
        writer = self.index.writer()
        for dataset_path in self.get_dataset_paths():
            with open(dataset_path) as csv_file:
                reader = csv.reader(csv_file, delimiter=',', quotechar='"')
                for item in reader:
                    label, path = check_paths(dataset_path, item[0], item[1])
                    writer.add_document(label=label, path=path)

        writer.commit()

    def get_dataset_paths(self):
        if self.config["search"]["dataset_paths"] == "test":
            return [test_dataset_path()]

        return [self.config["search"]["dataset_paths"]]

def test_dataset_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "test", "test-dataset.csv")

def check_paths(dataset_path, label, path):
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(dataset_path), path)

    assert os.path.exists(path),  "Path does not exist: " + path

    return label, path

def dedup(results_for_this_page, result_set):
    deduped_results = []

    for result in results_for_this_page:
        result_hash = get_hash(result)
        if not result_hash in result_set:
            deduped_results.append(result)

            result_set.add(result_hash)

    return deduped_results

def get_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

