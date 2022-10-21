import os
import requests
from tqdm import tqdm


home = os.getenv('HOME')

url = 'https://huggingface.co/datasets/MLCommons/peoples-speech-v2/resolve/main/train/clean/'
for i in tqdm(range(1000)):
    tar_number = i
    tar_number = (str(tar_number)).zfill(6)
    name = f'part-{tar_number}-5f4dc359-da2b-44e3-bc2e-acd03133d1f6-c000.tar'
    destination_filepath = f'{home}/data/the-peoples-speech/cc-by-clean/audios/{name}'
    file_path = url+f'clean_{tar_number}.tar'
    r = requests.get(file_path, stream=True)
    if r.status_code == 200:
        with open(destination_filepath, 'wb') as f:
            f.write(r.raw.read())
