
from search.engine.search_engine_factory import SearchEngineFactory

def search_for_data_book(config, data_book):
    return SearchEngineFactory(config).create().search_for_data_book(data_book)
