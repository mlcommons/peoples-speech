
from data_export.export.google_cloud_parallel_exporter import GoogleCloudParallelExporter
from data_export.export.local_exporter import LocalExporter
from data_export.export.cloud_exporter import CloudExporter

class ExporterFactory:
    def __init__(self, config):
        self.config = config

    def create(self):
        if self.config["exporter"]["type"] == "GoogleCloudParallelExporter":
            return GoogleCloudParallelExporter(self.config)
        if self.config["exporter"]["type"] == "LocalExporter":
            return LocalExporter(self.config)
        if self.config["exporter"]["type"] == "CloudExporter":
            return CloudExporter(self.config)

        assert False, "Unknown exporter type '" + self.config["exporter"]["type"] + "'"


