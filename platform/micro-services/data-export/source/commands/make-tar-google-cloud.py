
import concurrent.futures
from google.cloud import storage
from argparse import ArgumentParser
import logging
import csv
import os
import json
import tarfile
import io
from smart_open import open

logger = logging.getLogger(__name__)

storage_client = storage.Client()

def main():
    parser = ArgumentParser("Takes all of the files in a dataset and uploads into a TAR.GZ archive .")

    parser.add_argument("-i", "--input-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.5/train.csv",
        help = "The output path to load the dataset from.")
    parser.add_argument("-o", "--output-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.5/train.tar.gz",
        help = "The output path to save the test dataset.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    make_tar_gz(arguments)

def make_tar_gz(arguments):
    samples = load_csv(arguments["input_path"])

    updated_samples = update_samples(samples)

    writer = ArchiveWriter(arguments["output_path"], updated_samples)

    writer.run()

def load_csv(csv_path):
    new_samples = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')

        for row in reader:
            path, caption = row[0], row[1]

            metadata = {}
            if len(row) >= 3:
                metadata = json.loads(row[2])

            yield {"path" : path, "caption" : caption, "metadata" : metadata}

def update_samples(samples):
    for sample in samples:
        yield (update_path(sample["path"]), sample["path"], sample["caption"], sample["metadata"])

def update_path(path):
    bucket_name, prefix = get_bucket_and_prefix(path)
    return prefix

def get_bucket_and_prefix(path):
    parts = split_all(path[5:])

    assert len(parts) > 1, str(parts)

    return parts[0], os.path.join(*parts[1:])

def split_all(path):
    allparts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def open_archive(path):
    logger.debug("Opening archive for writing, " + path)
    tar_file = open(path, mode="wb", transport_params={"min_part_size" : (2**18 * 16)})
    return tarfile.TarFile(fileobj=tar_file, mode="w"), tar_file

class ArchiveWriter:
    def __init__(self, archive_path, samples):
        self.archive, self.archive_file = open_archive(archive_path)
        self.samples = samples

        self.csv_file_name = "data.csv"
        self.csv_file = open(self.csv_file_name, newline="", mode="w")
        self.csv_writer = csv.writer(self.csv_file, delimiter=',', quotechar='"')

    def run(self):

        sample_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:

            while True:
                sample_batch = self.get_next_batch()

                if len(sample_batch) == 0:
                    break

                # Start the load operations and mark each future with its URL
                future_to_data = {executor.submit(load_file, path): (updated_path, path, transcript, metadata) for updated_path, path, transcript, metadata in sample_batch}
                for future in concurrent.futures.as_completed(future_to_data):
                    updated_path, path, transcript, metadata = future_to_data[future]
                    try:
                        data = future.result()
                        info = tarfile.TarInfo(name=updated_path)
                        info.size = len(data)
                        data_buffer = io.BytesIO(data)
                        self.archive.addfile(info, data_buffer)
                        self.csv_writer.writerow([updated_path, transcript, json.dumps(metadata)])

                        logger.debug("loaded %s bytes from %s for sample %s " % (len(data), path, sample_count))

                        sample_count += 1

                        del data_buffer
                        del data
                        del info
                        del updated_path
                        del path
                        del transcript
                        del metadata
                    except Exception as exc:
                        print('%r generated an exception: %s' % (path, exc))


        self.csv_file.close()

        self.archive.add(self.csv_file_name)
        self.archive.close()
        self.archive_file.close()

    def get_next_batch(self):
        batch = []
        for i in range(1024):
            try:
                batch.append(next(self.samples))
            except StopIteration:
                break

        return batch


def load_file(path):
    bucket_name, prefix = get_bucket_and_prefix(path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(prefix)

    data = blob.download_as_bytes()

    return data


def setup_logger(arguments):

   if arguments["verbose"]:
       logger.setLevel(logging.DEBUG)
   else:
       logger.setLevel(logging.INFO)

   ch = logging.StreamHandler()
   ch.setLevel(logging.DEBUG)

   # create formatter
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

   # add formatter to ch
   ch.setFormatter(formatter)

   # add ch to logger
   logger.addHandler(ch)

if __name__ == "__main__":
    main()

