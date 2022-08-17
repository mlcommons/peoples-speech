from google.cloud import storage
from tqdm import tqdm
import random
import os

home = os.getenv('HOME')

bucket_name = 'the-peoples-speech-west-europe'

for i in tqdm(range(1000)):
    tar_number = i #random.randint(0,4450)
    tar_number = (str(tar_number)).zfill(5)
    tar_file = f'part-{tar_number}-5f4dc359-da2b-44e3-bc2e-acd03133d1f6-c000.tar'
    source_blob_name = f'forced-aligner/cuda-forced-aligner/peoples-speech/output_work_dir_9a/cc_by_clean/repartitioned_dataset_tars/{tar_file}'
    destination_file_name =f'{home}/data/the-peoples-speech/cc-by-clean/audios/{tar_file}'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


