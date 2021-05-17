
from data_export.utility.load_dataset_csv import load_dataset_csv
from data_export.utility.write_samples_to_tar_gz import write_samples_to_tar_gz

class LocalExporter:
    def __init__(self, config):
        self.config = config

    def export(self, dataset):
        samples = load_dataset_csv(dataset, task_id=0, task_count=1)

        path = self.config["output_dataset_path"]

        write_samples_to_tar_gz(samples, path)

