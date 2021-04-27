
from data_export.export.google_cloud_parallel_exporter import GoogleCloudParallelExporter

class ExporterFactory:
    def __init__(self, config):
        self.config = config

    def create(self):
        if self.config["exporter"]["type"] == "GoogleCloudParallelExporter":
            return GoogleCloudParallelExporter(self.config)

        assert False, "Unknown exporter type '" + self.config["exporter"]["type"] + "'"


