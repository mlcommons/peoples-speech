
from data_export.google.google_cloud_work_queue import GoogleCloudWorkQueue
import json

class GoogleCloudParallelExporter:
    def __init__(self, config):
        self.config = config

        self.work_queue = GoogleCloudWorkQueue(config)

    def export(self, output_dataset_path, dataset):
        print("Exporting dataset", dataset)

        task_count = int(self.config["exporter"]["task_count"])

        for task_id in range(task_count):
            self.work_queue.push(json.dumps({
                "task_id" : task_id,
                "task_count" : task_count,
                "dataset" : dataset}))

        self.start_job()

    def start_job(self):
        kubernetes_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "build-scripts", "kubernetes")
        command = "kubectl apply -f run-export-worker.yaml"
        subprocess.run(command, cwd = kubernetes_directory)



