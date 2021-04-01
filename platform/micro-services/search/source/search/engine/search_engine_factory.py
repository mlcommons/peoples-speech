
from search.engine.existing_dataset_search_engine import ExistingDatasetSearchEngine

class SearchEngineFactory:
    def __init__(self, config):
        self.config = config

    def create(self):
        if self.config["search"]["engine"] == "ExistingDatasetSearchEngine":
            return ExistingDatasetSearchEngine(self.config)

        assert False, "Could not find search engine " + self.config["search"]["engine"]


