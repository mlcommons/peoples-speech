
import os
import shutil
import subprocess
import csv
import json

def write_samples_to_tar_gz(samples, path):
    os.makedirs("/tmp/export-data", exist_ok=True)

    # copy locally
    command = ["gsutil", "-m", "cp", "-I", "/tmp/export-data"]

    stream_to_command(command, samples)

    # tar it
    filename = os.path.basename(path)

    temp_archive = os.path.join("/tmp", filename)

    command = ["tar", "-czvf", temp_archive, "export-data"]

    subprocess.run(command, cwd="/tmp")

    # upload it
    command = ["gsutil", "-m", "cp", temp_archive, path]

    subprocess.run(command)

def stream_to_command(command, samples):
    p = subprocess.Popen(command, stdin=subprocess.PIPE)
    with open("/tmp/export-data/dataset.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')

        for sample in samples:
            writer.writerow([sample["path"], sample["caption"], json.dumps(sample["metadata"])])
            line = sample["path"] + "\n"
            try:
                p.stdin.write(line.encode('utf-8'))
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
