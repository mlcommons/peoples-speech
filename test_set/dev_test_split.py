import os
import random
import argparse
from distutils import dir_util

from test_set import data_classes

SPLIT_MANIFEST_PATH = "{out_dir}/{split}/manifest.jsonl"
SPLIT_AUDIO_DIR = "{out_dir}/{split}/training-audio"

def main():
    parser = argparse.ArgumentParser(
        description="Split a TPS formatted dataset in two"
    )
    parser.add_argument(
        "--manifest_path",
        required=True,
        type=str,
        help="Path to JSONL file. Each line describes an object of class"
             "test_data.data_classes.AudioData"
    )
    parser.add_argument(
        "--audio_dir",
        required=True,
        type=str,
        help="Training audio directory"
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Store split datasets here"
    )
    args = parser.parse_args()
    test_file_args = {"out_dir": args.out_dir, "split": "test"}
    test_manifest_path = SPLIT_MANIFEST_PATH.format(**test_file_args)
    test_audio_dir = SPLIT_AUDIO_DIR.format(**test_file_args)
    os.makedirs(test_audio_dir)
    dev_file_args = {"out_dir": args.out_dir, "split": "dev"}
    dev_manifest_path = SPLIT_MANIFEST_PATH.format(**dev_file_args)
    dev_audio_dir = SPLIT_AUDIO_DIR.format(**dev_file_args)
    os.makedirs(dev_audio_dir)
    with open(args.manifest_path, "r") as manifest, \
         open(dev_manifest_path, "w") as dev_manifest, \
         open(test_manifest_path, "w") as test_manifest:
        
        # Count total audio time in input manifest
        print("Counting total audio time in input manifest")
        total_ms = 0
        for json_line in manifest:
            audio_data = data_classes.AudioData.from_json_str(json_line)
            total_ms += audio_data.get_total_duration_ms()
        total_hours_str = "{:.2f}".format(total_ms / (1000 * 3600))
        print(f"Found {total_hours_str}h of audio in total")

        # Split into dev and test with similar total duration
        print("Creating dev and test splits")
        total_dev_ms = 0
        total_test_ms = 0
        manifest.seek(0)
        for json_line in manifest:
            audio_data = data_classes.AudioData.from_json_str(json_line)
            ms_in_line = audio_data.get_total_duration_ms()
            if total_dev_ms > total_ms / 2:
                write_to_dev = True
            elif total_test_ms > total_ms / 2:
                write_to_dev = False
            else:
                write_to_dev = random.random() > 0.5

            if write_to_dev:
                dev_manifest.write(json_line)
                dir_util.copy_tree(
                    os.path.join(args.audio_dir, audio_data.identifier),
                    os.path.join(dev_audio_dir, audio_data.identifier)
                )
                total_dev_ms += ms_in_line
            else:
                test_manifest.write(json_line)
                dir_util.copy_tree(
                    os.path.join(args.audio_dir, audio_data.identifier),
                    os.path.join(test_audio_dir, audio_data.identifier)
                )
                total_test_ms += ms_in_line
        dev_hours_str = "{:.2f}".format(total_dev_ms / (1000 * 3600))
        print(f"Dev set totals {dev_hours_str}h")
        test_hours_str = "{:.2f}".format(total_test_ms / (1000 * 3600))
        print(f"Test set totals {test_hours_str}h")
if __name__ == "__main__":
    main()