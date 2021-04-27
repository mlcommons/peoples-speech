
from data_export.export import ExporterFactory
from data_export.database import get_dataset_from_id
from data_export.utility.get_config import get_config

def export_dataset_by_id(dataset_id, config = get_config()):
    exporter = ExporterFactory(config).create()
    dataset = get_dataset_from_id(config, dataset_id)
    exporter.export(dataset)

