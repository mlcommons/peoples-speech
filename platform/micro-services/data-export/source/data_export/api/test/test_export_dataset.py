import unittest
import tarfile
import io
from data_export.api.export_dataset import export_dataset
from data_export.utility.load_dataset_csv import load_dataset_csv
from data_export.utility.load_dataset_csv import load_dataset_csv_from_file

class TestDataExport(unittest.TestCase):

    def test_tiny_local(self):
        tiny_dataset = "gs://the-peoples-speech-west-europe/peoples-speech-v0.8/unittest.csv"
        tiny_config = {
            "exporter" : { "type" : "LocalExporter" } }

        export_dataset("/tmp/export-data/unittest.tar.gz", tiny_dataset, tiny_config)

        self.extract_dataset(tiny_config)
        self.verify_dataset(tiny_dataset, tiny_config)

    def extract_dataset(self, config):
        with tarfile.open(config["output_dataset_path"], "r:gz") as tar:
            members = tar.getmembers()

    def verify_dataset(self, dataset, config):

        samples = load_dataset_csv(dataset, task_id=0, task_count=1)
        with tarfile.open(config["output_dataset_path"], "r:gz") as tar:
            samples_file = io.TextIOWrapper(tar.extractfile("export-data/dataset.csv"))
            loaded_samples = load_dataset_csv_from_file(samples_file, task_id=0, task_count=1)

            for original_sample, new_sample in zip(samples, loaded_samples):
                self.assertEqual(original_sample, new_sample)


if __name__ == '__main__':
    unittest.main()
