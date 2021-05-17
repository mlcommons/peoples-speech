
from data_export.google.google_cloud_work_queue import GoogleCloudWorkQueue
from data_export.utility.get_config import get_config
from data_export.utility.load_dataset_csv import load_dataset_csv

import json

def main():
    config = get_config()
    q = GoogleCloudWorkQueue(config)
    print("Worker with sessionID: " +  q.sessionID())
    print("Initial queue state: empty=" + str(q.empty()))
    while not q.empty():
        item = q.lease(lease_secs=3600, block=True, timeout=2)
        if item is not None:
            itemstr = item.decode("utf-8")
            print("Working on " + itemstr)
            export_data(itemstr, config)
            q.complete(item)
        else:
            print("Waiting for work")
    print("Queue empty, exiting")

def export_data(itemstr, config):
    item = json.loads(itemstr)

    task_id = item["task_id"]
    task_count = item["task_count"]
    dataset = item["dataset"]

    samples = load_csv(dataset, task_id, task_count)

    path = config["exporter"]["output_path"] + "-" + str(task_id) + "-tar.gz"

    write_samples_to_tar_gz(samples, path)

if __name__ == "__main__":
    main()


