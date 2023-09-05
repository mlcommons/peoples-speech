import json
from pathlib import Path
import os
import glob
import re
import regex
import sox
import tqdm


SRC_DATA_DIR = "/home/rafael/supervised_peoples_speech/instructions_for_aligning_peoples_speech_test_set/test_dataset_April_18_2023/data"
TGT_MANIFEST = "manifest.json"

SEPARATOR = "<segment_split>"


def is_timestamp_line(line):
    TIMESTAMP_REGEX = "^[\d.:,]+ --> [\d.:,]+$"
    if re.match(TIMESTAMP_REGEX, line):
        return True
    return False


def add_segment_split_to_text(text, segment_separator):
    # remove some symbols for better split into sentences
    text = (
        text.replace("\n", " ")
        .replace("\t", " ")
        .replace("…", "...")
        .replace("\\", " ")
        .replace("--", " -- ")
        .replace(". . .", "...")
    )

    # end of quoted speech - to be able to split sentences by full stop
    text = re.sub(r"([\.\?\!])([\"\'])", r"\g<2>\g<1> ", text)

    # remove extra space
    text = re.sub(r" +", " ", text)

    # remove normal brackets, square brackets and curly brackets
    text = re.sub(r"(\(.*?\))", " ", text)
    text = re.sub(r"(\[.*?\])", " ", text)
    text = re.sub(r"(\{.*?\})", " ", text)

    # remove space in the middle of the lower case abbreviation to avoid splitting into separate sentences
    matches = re.findall(r"[a-z]\.\s[a-z]\.", text)
    for match in matches:
        text = text.replace(match, match.replace(". ", "."))

    # find phrases in quotes
    with_quotes = re.finditer(r"“[A-Za-z ?]+.*?”", text)
    sentences = []
    last_idx = 0
    for m in with_quotes:
        match = m.group()
        match_idx = m.start()
        if last_idx < match_idx:
            sentences.append(text[last_idx:match_idx])
        sentences.append(match)
        last_idx = m.end()
    sentences.append(text[last_idx:])
    sentences = [s.strip() for s in sentences if s.strip()]

    # Read and split text by utterance (roughly, sentences)
    split_pattern = (
        f"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!|\.”|\?”\!”)\s"
    )

    new_sentences = []
    for sent in sentences:
        new_sentences.extend(regex.split(split_pattern, sent))
    sentences = [s.strip() for s in new_sentences if s.strip()]

    sentences = [" ".join(sent.split()) for sent in sentences]

    # remove any "- " at start of sentences
    sentences = [re.sub(r"^- ", "", sent) for sent in sentences]

    text = segment_separator.join(sentences)

    return text


def get_text_from_srt_file(filepath):
    with open(filepath, "r", encoding="utf-8-sig") as fin:
        # get all lines, without newline char
        lines = [line.strip() for line in fin.readlines()]

    # remove the lines that are just numbers and which come before the timestamp lines
    # and the timestamp lines themselves
    lines_new = []
    for line_i, line in enumerate(lines):
        if line_i < len(lines) - 1:
            if (not is_timestamp_line(lines[line_i + 1])) and (
                not is_timestamp_line(line)
            ):
                lines_new.append(line)

    lines = lines_new

    # merge all lines into one
    text = " ".join(lines)

    text = add_segment_split_to_text(text, segment_separator=SEPARATOR)

    return text


utt_ids = glob.glob(SRC_DATA_DIR + "/*.flac")
utt_ids = [os.path.basename(u).split(".flac")[0] for u in utt_ids]
print(len(utt_ids), "Utt IDs")
n_segments = 0

with open(TGT_MANIFEST, "w", encoding="utf8") as fout:
    for utt_id in tqdm.tqdm(utt_ids):
        print(utt_id, "Utt ID")
        srt_file = os.path.join(SRC_DATA_DIR, f"{utt_id}.srt")
        flac_file = os.path.join(SRC_DATA_DIR, f"{utt_id}.flac")
        tgt_wav_file = os.path.join(SRC_DATA_DIR, "wavs", f"{utt_id}.wav")
        print("This is the tgt wav", tgt_wav_file)

        text = get_text_from_srt_file(srt_file)

        n_segments += len(text.split(SEPARATOR))

        data = {
            "audio_filepath": flac_file,
            "duration": float(sox.file_info.duration(flac_file)),
            "text": text,
        }
        print(data)

        fout.write(f"{json.dumps(data)}\n")

print("total number of segments:", n_segments)
