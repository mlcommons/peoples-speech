
from data_book.importer.fast_text_importer import FastTextImporter

class ImporterFactory:
    def __init__(self, config):
        self.config = config

    def create(self):

        assert self.config["import"] == "fasttext"

        return FastTextImporter(self.config)




