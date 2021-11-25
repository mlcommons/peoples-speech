import os
import json
import argparse
from tqdm import tqdm

import srt
from pydub import AudioSegment

from test_set import data_classes
from test_set import text_preprocessing

AUDIO_PATH = "{audio_dir}/{identifier}/{audio_document_id}"
AUDIO_SEGMENT_PATH = "{audio_dir}/{identifier}/{audio_segment_name}"
SUBRIP_PATH = "{subrip_dir}/{identifier}/{text_document_id}"
OUTPUT_MANIFEST_PATH = "{out_dir}/manifest.jsonl"
OUTPUT_AUDIO_DIR = "{out_dir}/training-audio"
AUDIO_FORMAT = "flac"
SENTENCE_END_CHARS = set(["!", ".", "?"])

def get_audio_name(audio_document_id):
    # Remove "." because it has special meaning in webdataset format
    # Remove " " because kaldi keys may not contain " "
    # name = name.replace("/", "_SLASH_")
    name = audio_document_id.replace(".", "_DOT_")
    name = name.replace(" ", "_SPACE_")
    return name

def get_audio_segment_name(audio_name, index, suffix):
    return f"{audio_name}_{index:05d}.{suffix}"

def main():
    parser = argparse.ArgumentParser(
        description="Parse an SRT transcription into a NeMo-style dataset"
    )
    parser.add_argument(
        "--manifest_path",
        required=True,
        type=str,
        help="Path to JSONL file. Each line has keys ['identifier',"
             "'audio_document_id', 'text_document_id']"
    )
    parser.add_argument(
        "--audio_dir",
        required=True,
        type=str,
        help="Directory containing {identifier}/{audio_document_id}"
    )
    parser.add_argument(
        "--subrip_dir",
        required=True,
        type=str,
        help="Directory containing {identifier}/{text_document_id}"
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Path for the output manifest and chunked audios"
    )
    args = parser.parse_args()
    output_manifest_path = OUTPUT_MANIFEST_PATH.format(out_dir=args.out_dir)
    output_audio_dir = OUTPUT_AUDIO_DIR.format(out_dir=args.out_dir)
    os.makedirs(output_audio_dir)
    with open(args.manifest_path, "r") as input_manifest, \
         open(output_manifest_path, "w") as output_manifest:
        for str_audio_data in input_manifest:
            # Parse manifest line
            audio_data = data_classes.AudioData.from_json_str(str_audio_data)
            audio_name = get_audio_name(audio_data.audio_document_id)
            
            # Assert that files for this line exist
            assert audio_data.audio_document_id.endswith("." + AUDIO_FORMAT), \
                f"Found non-{AUDIO_FORMAT} audio_document_id: {str_audio_data}"
            audio_path = AUDIO_PATH.format(
                audio_dir=args.audio_dir,
                identifier=audio_data.identifier,
                audio_document_id=audio_data.audio_document_id
            )
            assert os.path.exists(audio_path), \
                f"Audio file not found: {audio_path}"
            subrip_path = SUBRIP_PATH.format(
                subrip_dir=args.subrip_dir,
                identifier=audio_data.identifier,
                text_document_id=audio_data.text_document_id
            )
            assert os.path.exists(subrip_path), \
                f"Subrip file not found: {subrip_path}"
            
            # Read audio and transcription
            full_audio = AudioSegment.from_file(audio_path)
            full_audio = full_audio.set_frame_rate(16000)
            full_audio = full_audio.set_channels(1)
            with open(subrip_path, "r") as subrip_file:
                subrip_string = subrip_file.read()
            subtitles = srt.parse(subrip_string)
            
            # Write sentences as training data
            current_sentence = None
            start_ms = 0
            sentence_idx = 0
            for subtitle in tqdm(subtitles):
                unquoted_text = subtitle.content.replace("\"", "")
                stripped_unquoted_text = unquoted_text.strip()
                sentence_end = stripped_unquoted_text[-1] in SENTENCE_END_CHARS
                clean_text = text_preprocessing.clean_text(stripped_unquoted_text)
                if len(clean_text) == 0:
                    continue
                if current_sentence is None:
                    start_ms = round(subtitle.start.total_seconds() * 1000)
                    current_sentence = clean_text
                else:
                    current_sentence += " " + clean_text
                if sentence_end:
                    end_ms = round(subtitle.end.total_seconds() * 1000)
                    audio_segment = full_audio[start_ms:end_ms]
                    audio_segment_name = get_audio_segment_name(
                        audio_name=audio_name,
                        index=sentence_idx,
                        suffix=AUDIO_FORMAT
                    )
                    audio_segment_path = AUDIO_SEGMENT_PATH.format(
                        audio_dir=output_audio_dir,
                        identifier=audio_data.identifier,
                        audio_segment_name=audio_segment_name
                    )
                    audio_segment_relpath = os.path.relpath(
                        audio_segment_path,
                        output_audio_dir
                    )
                    os.makedirs(os.path.dirname(audio_segment_path), exist_ok=True)
                    audio_segment.export(audio_segment_path, format=AUDIO_FORMAT)
                    current_sentence = text_preprocessing.clean_text(current_sentence)
                    audio_data.append_training_data(
                        duration_ms=end_ms - start_ms,
                        label=current_sentence,
                        name=audio_segment_relpath
                    )
                    current_sentence = None
                    sentence_idx += 1
            str_audio_data = json.dumps(audio_data.to_dict())
            output_manifest.write(str_audio_data + "\n")

if __name__ == "__main__":
    main()