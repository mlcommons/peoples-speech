
from task_manager.engine.engine_factory import EngineFactory

from search.api import search_for_data_book
from ingestion.api import ingest_search_hits
from forced_alignment.api import align_dataset
from quality.api import clean_dataset

def make_dataset(config, data_book):
    engine = EngineFactory(config).create()

    # search
    hits = engine.run(search_for_data_book, config, data_book)

    # ingest
    long_dataset = engine.run(ingest_search_hits, config, hits)

    # align
    aligned_dataset = engine.run(align_dataset, config, long_dataset)

    # clean
    cleaned_dataset = engine.run(clean_dataset, config, aligned_dataset)

    return cleaned_dataset

def run_search_task(engine, config, data_book):
    return engine.run(search_for_data_book, config, data_book)





