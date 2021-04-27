import time

from data_export.google.google_cloud_work_queue import GoogleCloudWorkQueue
from data_export.utility.get_config import get_config

from smart_open import open
import csv
import json

import subprocess
import errno

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

    write_samples(samples, path)

def write_samples(samples, path):
    # copy locally
    command = ["gsutil", "-m", "cp", "-I", "/tmp/export-data"]

    stream_to_command(command, samples)

    # tar it
    filename = os.path.basename(path)

    temp_archive = os.path.join("/tmp", filename)

    command = ["tar", "-czvf", temp_archive, "/tmp/export-data"]

    subprocess.run(command)

    # upload it
    command = ["gsutil", "-m", "cp", temp_archive, path]

def stream_to_command(command, samples):
    p = subprocess.Popen(command, stdin=subprocess.PIPE)
    for sample in samples:
        line = sample[0]
        try:
            p.stdin.write(line)
        except IOError as e:
            if e.errno == errno.EPIPE or e.errno == errno.EINVAL:
                # Stop loop on "Invalid pipe" or "Invalid argument".
                # No sense in continuing with broken pipe.
                break
            else:
                # Raise any other error.
                raise

    p.stdin.close()
    p.wait()

def load_csv(csv_path, task_id, task_count):

    index = 0
    new_samples = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')

        for row in reader:
            path, caption = row[0], row[1]

            metadata = {}
            if len(row) >= 3:
                metadata = json.loads(row[2])

            if index % task_count == task_id:
                yield {"path" : path, "caption" : caption, "metadata" : metadata}

            index += 1

if __name__ == "__main__":
    main()


