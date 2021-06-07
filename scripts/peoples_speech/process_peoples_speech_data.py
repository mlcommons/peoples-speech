# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
import subprocess
from typing import Dict

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Downloads and processes MLCommons People\'s Speech dataset.')
parser.add_argument("--data_root", type=str, required=True,
                    help="Directory containing the audio dataset.")
parser.add_argument("--data_manifest", type=str, required=True,
                    help="Json lines file containing the audio paths and associated transcripts")
parser.add_argument('--dest_root', type=str, required=True,
                    help='Output directory for wav data and manifest')
parser.add_argument("--num_workers", default=multiprocessing.cpu_count(),
                    type=int, help="Workers to process dataset.")
parser.add_argument('--sample_rate', default=8000, type=int, help='Sample rate')
args = parser.parse_args()

def convert_to_wav(input_manifest_json: Dict):
    output_manifest_jsons = []
    training_samples = input_manifest_json["training_data"]
    for label, audio_path, duration in zip(training_samples["label"],
                                           training_samples["output_paths"],
                                           training_samples["duration_ms"]):
        base_name, _ = os.path.splitext(audio_path)
        base_name += ".wav"
        subdir, _ = os.path.split(base_name)
        os.makedirs(os.path.join(args.dest_root, subdir), exist_ok=True)
        output_wav_path = os.path.join(args.dest_root, base_name)
        input_path = os.path.join(args.data_root, audio_path)
        subprocess.check_call(["sox", input_path, "-t", "wav", "--channels", "1", "--rate", str(args.sample_rate), "--encoding", "signed", "--bits", "16", output_wav_path])
        output_manifest_jsons.append({"audio_filepath": output_wav_path,
                                      "text": label,
                                      "duration": duration})
    return output_manifest_jsons

def main():
    os.makedirs(args.dest_root, exist_ok=True)
    input_manifest_jsons = []
    with open(args.data_manifest) as fh:
        for line in fh:
            input_manifest_jsons.append(json.loads(line))
    with ThreadPoolExecutor(args.num_workers) as executor:
        json_objs = list(tqdm(executor.map(convert_to_wav, input_manifest_jsons),
                              total=len(input_manifest_jsons)))
        manifest_lines = [item for sublist in json_objs for item in sublist]
    with open(os.path.join(args.dest_root, "manifest.json"), "w") as fh:
        for json_obj in manifest_lines:
            json.dump(json_obj, fh)
            fh.write("\n")

if __name__ == '__main__':
    main()
