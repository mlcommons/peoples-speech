import json
import argparse
from tqdm import tqdm

import librosa

def main():
    parser = argparse.ArgumentParser(
        description="Append transcriptions to a dataset manifest, including a "
            "few similarity metrics wrt the reference text"
    )
    parser.add_argument(
        "--untarred_manifest_path",
        required=True,
        type=str,
        help="NeMo-style manifest with local, untarred audio filepaths"
    )
    parser.add_argument(
        "--output_manifest_path",
        required=True,
        type=str,
        help="NeMo-style manifest with corrected audio file durations"
    )
    args = parser.parse_args()
    with open(args.untarred_manifest_path, "r") as input_manifest, \
         open(args.output_manifest_path, "w") as output_manifest:
        for str_audio_sample in tqdm(input_manifest):
            audio_sample = json.loads(str_audio_sample)
            audio_filepath = audio_sample["audio_filepath"]
            audio_sample["duration"] = librosa.get_duration(filename=audio_filepath)
            output_manifest.write(json.dumps(audio_sample) + "\n")

if __name__ == "__main__":
    main()