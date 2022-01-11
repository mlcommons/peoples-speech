import os
import glob
import json
import math
import random
import tarfile
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Subset a tarred local dataset"
    )
    parser.add_argument(
        "--dataset_tarfiles",
        required=True,
        type=str,
        help="A regex of local .tar files"
    )
    parser.add_argument(
        "--dataset_manifest",
        required=True,
        type=str,
        help="TPS-style manifest of the large dataset"
    )
    parser.add_argument(
        "--subset_rel_size",
        required=True,
        type=float,
        help="Subset size relative to the large dataset's (e.g. 0.1)"
    )
    parser.add_argument(
        "--tarfiles_frac",
        required=True,
        type=float,
        help="Fraction of all tarfiles to browse while subsetting"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Name of the output tarfile"
    )
    args = parser.parse_args()
    assert args.tarfiles_frac >= args.subset_rel_size
    # Build subset tarfile
    print("Building subset tarfile")
    os.makedirs(args.output_dir)
    output_tar_path = os.path.join(args.output_dir, "subset.tar")
    input_tar_paths = glob.glob(args.dataset_tarfiles)
    total_tarfiles = len(input_tar_paths)
    input_tar_paths = random.sample(
        input_tar_paths,
        math.ceil(args.tarfiles_frac * total_tarfiles)
    )
    real_tarfiles_frac = len(input_tar_paths) / total_tarfiles
    keeper_names = set()
    with tarfile.open(output_tar_path, "w") as output:
        for input_tar_path in tqdm(input_tar_paths):
            with tarfile.open(input_tar_path, "r:") as input_tar_file:
                members = input_tar_file.getmembers()
                n_keepers = len(members) * args.subset_rel_size / real_tarfiles_frac
                if n_keepers < 1:
                    n_keepers = 1 if (random.random() < n_keepers) else 0
                else:
                    n_keepers = int(n_keepers)
                keepers = random.sample(members, n_keepers)
                for keeper in keepers:
                    output.addfile(keeper, input_tar_file.extractfile(keeper.name))
                    keeper_names.add(keeper.name)
    # Build subset manifest
    print("Building subset manifest")
    subset_manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    with open(args.dataset_manifest, "r") as full_manifest, \
         open(subset_manifest_path, "w") as subset_manifest:
        for str_sample in tqdm(full_manifest):
            sample = json.loads(str_sample)
            if sample["audio_filepath"] in keeper_names:
                subset_manifest.write(str_sample)

if __name__ == "__main__":
    main()