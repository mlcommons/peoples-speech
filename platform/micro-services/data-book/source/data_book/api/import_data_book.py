
from data_book.importer.importer_factory import ImporterFactory

def import_data_book(config):
    ImporterFactory(config).create().import_data_book()
