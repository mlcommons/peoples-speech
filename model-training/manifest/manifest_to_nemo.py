import argparse
from tqdm import tqdm

from manifest import data_classes

def main():
    parser = argparse.ArgumentParser(
        description="Form a NeMo manifest from a TPS-formatted manifest"
    )
    parser.add_argument(
        "--tps_manifest_path",
        required=True,
        type=str,
        help="Path to JSONL file. Each line has keys ['identifier',"
             "'audio_document_id', 'text_document_id', 'training_data']"
    )
    parser.add_argument(
        "--nemo_manifest_path",
        required=True,
        type=str,
        help="Output NeMo manifest path"
    )
    parser.add_argument(
        "--audio_dir",
        required=False,
        type=str,
        help="Provide if you need absolute paths in the output NeMo manifest"
    )
    args = parser.parse_args()
    with open(args.tps_manifest_path, "r") as tps_manifest, \
         open(args.nemo_manifest_path, "w") as nemo_manifest:
        for tps_manifest_line in tqdm(tps_manifest):
            audio_data = data_classes.AudioData.from_json_str(tps_manifest_line)
            nemo_lines = audio_data.to_nemo_lines(audio_dir=args.audio_dir)
            for nemo_line in nemo_lines:
                nemo_manifest.write(nemo_line + "\n")

if __name__ == "__main__":
    main()