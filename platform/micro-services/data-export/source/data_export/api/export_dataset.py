
from data_export.export import ExporterFactory
from data_export.utility.get_config import get_config

def export_dataset(output_dataset, dataset, config = get_config()):
    exporter = ExporterFactory(config).create()
    return exporter.export(output_dataset, dataset)
