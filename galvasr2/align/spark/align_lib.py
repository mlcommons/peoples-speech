from functools import partial
from io import BytesIO
import json
import shlex
import subprocess
from typing import List, Tuple
import wave

import ds_ctcdecoder
from ftfy import fix_text, guess_bytes
import langid
import numpy as np
import pandas as pd

import re
import srt

from pyspark.sql.functions import col, pandas_udf
import pyspark.sql.types as T
from pyspark.sql.functions import array, array_contains, count, explode, lit
from pyspark.sql.types import (ArrayType, BinaryType, DoubleType, FloatType,
                               ShortType, StructType, StructField, StringType,
                               IntegerType, LongType)


def DecodeToWavPipe(input_bytes, fmt):
  cmd = f'sox -t {fmt} - -t wav --channels 1 --rate 16000 --encoding signed --bits 16 -'
  p = subprocess.Popen(shlex.split(cmd),
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
  out, err = p.communicate(input=input_bytes)
  assert p.returncode == 0, err
  return out

@pandas_udf(StringType())
def srt_to_text(srt_file_contents: pd.Series) -> pd.Series:
  def helper(content: str) -> str:
    try:
      return " ".join(line.content.replace("\n", " ") for line in srt.parse(content))
    except (srt.SRTParseError, srt.TimestampParseError) as exc:
      # Is this really the best way to log in a pandas UDF?
      print("WARNING: trouble parsing content")
      print(exc)
      return ""
  return srt_file_contents.apply(helper)

@pandas_udf(StringType())
def infer_language_udf(text_column: pd.Series) -> pd.Series:
  return text_column.apply(lambda string: langid.classify(string)[0] if string else "")


# bytes, length, sampling_frequency, number_channels
def load_audio_files(spark, collected_audio_document_rows, base_path: str):
  audio_document_ids = [os.path.join(base_path, row.identifier, row.audio_document_id)
                        for row in collected_audio_document_rows]
  raw_audio_df = (spark.read.format("binaryFile")
                  .load(audio_document_ids))

  return raw_audio_df.select('content',
                             F.reverse(F.split(raw_audio_df.path, "[.]"))[0].alias("format"),
                             F.reverse(F.split(raw_audio_df.path, "/"))[0].alias("audio_document_id"),
                             F.monotonically_increasing_id().alias("int64_audio_document_id")
  )


@pandas_udf(StringType())
def fix_text_udf(binary_column: pd.Series) -> pd.Series:
  return binary_column.apply(lambda b: fix_text(guess_bytes(b)[0]))

# https://spark.apache.org/docs/3.0.2/api/python/pyspark.sql.html#pyspark.sql.HiveContext.newSession
def load_transcripts(spark, base_path: str, collected_text_document_rows: List[pyspark.Row]):
  text_document_ids = [os.path.join(base_path, row.identifier, row.text_document_id)
                       for row in collected_text_document_rows]
  srt_df = (spark.read.format("binaryFile")
            .load(text_document_ids))
  # Note the duplication with load_audio_files
  return srt_df.select(srt_to_text(fix_text_udf(srt_df.content)).alias('transcript'),
                       F.reverse(F.split(srt_df.path, "/"))[0].alias("text_document_id"),
                       #F.reverse(F.split(srt_df.path, "/"))[1].alias("identifier")
  )


def prepare_vad_udf(num_padding_frames, threshold, aggressiveness, frame_duration_ms):
  # Each audio file returns multiple voiced fragments. I need an Array, don't I?
  return_type = StructType(
    [
      StructField("start_ms", ArrayType(IntegerType())),
      StructField("end_ms", ArrayType(IntegerType())),
      StructField("voiced_buffer", ArrayType(BinaryType())),
    ]
  )
  # Try using ArrayType(BinaryType()). Need to convert numpy array to bytearray
  # Need a java UDF to reinterpet bytes, it seems https://stackoverflow.com/a/57848517
  # Or I could just use np.ndarray.view(np.int8) right here.
  AUDIO_FORMAT = AudioFormat(sample_rate=16_000, channels=1, sample_byte_width=2)
  FRAME_DURATION_SAMPLES = (AUDIO_FORMAT.sample_rate * frame_duration_ms) // 1000
  FRAME_DURATION_BYTES = (FRAME_DURATION_SAMPLES * AUDIO_FORMAT.channels *
                          AUDIO_FORMAT.sample_byte_width)
  @pandas_udf(return_type)
  def vad(audio_series: pd.Series, audio_types_series: pd.Series, audio_document_id_series: pd.Series) -> pd.DataFrame:
    df_rows = []
    for audio_buffer, audio_type, audio_document_id in zip(audio_series,
                                                           audio_types_series,
                                                           audio_document_id_series):
      wav_bytes_buffer = BytesIO(DecodeToWavPipe(audio_buffer, audio_type))
      with wave.open(wav_bytes_buffer, "rb") as fh:
        num_frames = fh.getnframes()
        assert fh.getframerate() == AUDIO_FORMAT.sample_rate
        assert fh.getnchannels() == AUDIO_FORMAT.channels
        assert fh.getsampwidth() == AUDIO_FORMAT.sample_byte_width
        pcm_buffer = fh.readframes(num_frames)
        del wav_bytes_buffer
        num_frames = len(pcm_buffer) // FRAME_DURATION_BYTES
        buffers = [pcm_buffer[FRAME_DURATION_BYTES * i: FRAME_DURATION_BYTES * (i + 1)] for i in range(num_frames)]
        del pcm_buffer
        generator = vad_split(buffers, AUDIO_FORMAT, num_padding_frames, 
                              threshold, aggressiveness)
        
        voiced_buffer_list, start_ms_list, end_ms_list = [], [], []
        total_serialized_bytes = 0
        for voiced_buffer, start_ms, end_ms in generator:
          total_serialized_bytes += 2 * len(voiced_buffer)
          if total_serialized_bytes > 2 * 1024 * 1024 * 1024 - 1024 * 1024 * 1024:
            two_sum = lambda x, y: (sum(x), sum(y))
            ignored_bytes = 0
            ignored_ms = 0.0
            for voiced_buffer, start_ms, end_ms in generator:
              ignored_bytes += len(voiced_buffer)
              ignored_ms += (end_ms - start_ms)
            ignored_gigabytes = ((ignored_bytes / 1024) / 1024) / 1024
            ignored_hours = ((ignored_ms / 1000) / 60) / 60
            print(f"WARNING: truncating voice-activity-detected audio to less than 2GB for {audio_document_id}. Wasted {ignored_gigabytes}GB of data. Wasted {ignored_hours} hours of data.")
            break
          voiced_buffer_list.append(voiced_buffer)
          start_ms_list.append(start_ms)
          end_ms_list.append(end_ms)
        del buffers
        # mb_total = sum(voiced_buffer.nbytes / 1024 / 1024 for voiced_buffer in voiced_buffer_list)
        # print("GALVEZ: Chunk size in MB: ", mb_total)
        df_rows.append({"start_ms": start_ms_list,
                        "end_ms": end_ms_list,
                        "voiced_buffer": voiced_buffer_list})
    return pd.DataFrame(df_rows)
  return vad

GENERATE_LM_OUTPUT_SCHEMA=StructType([StructField("path", StringType())])
def prepare_generate_lm_udf(kenlm_path: str, debug_work_dir: str, alphabet_path: str):
  # @pandas_udf(GENERATE_LM_OUTPUT_SCHEMA)
  # TODO: Need to sort the log_probabilities by int64_uttid (right?)
  def generate_lm(grouping_key: Tuple[np.str, np.str],
                  data_df: pd.DataFrame) -> pd.DataFrame:
    identifier, text_document_id, = grouping_key
    identifier = str(identifier)
    text_document_id = str(text_document_id)

    transcript = data_df.transcript[0]
    with tempfile.NamedTemporaryFile('w+t', dir=debug_work_dir) as input_txt:
      input_txt.write(transcript)
      input_txt.flush()
      os.makedirs(os.path.join(debug_work_dir, identifier), exist_ok=True)
      scorer_path = os.path.join(debug_work_dir, identifier, text_document_id + ".scorer")
      data_lower, vocab_str = convert_and_filter_topk(scorer_path, input_txt.name, 500000)
      build_lm(scorer_path, kenlm_path, 5, '85%', '0|0|1', True, 255, 8,
               'trie', data_lower, vocab_str)
      os.remove(scorer_path + '.' + 'lower.txt.gz')
      os.remove(scorer_path + '.' + 'lm.arpa')
      os.remove(scorer_path + '.' + 'lm_filtered.arpa')

      create_bundle(alphabet_path, scorer_path + '.' + 'lm.binary',
                    scorer_path + '.' + 'vocab-500000.txt',
                    scorer_path,
                    False, 0.931289039105002, 1.1834137581510284)
      os.remove(scorer_path + '.' + 'lm.binary')
      os.remove(scorer_path + '.' + 'vocab-500000.txt')

    with open(alphabet_path) as fh:
      num_output_symbols = len(fh.readlines()) + 1
    assert num_output_symbols == 32, f"GALVEZ:{num_output_symbols}"
    transcripts = []

    id_to_symbol = {}
    with open(alphabet_path) as fh:
      for i, line in enumerate(fh):
        id_to_symbol[i] = line.rstrip()
    id_to_symbol[31] = "blank"

    for row in data_df.itertuples():
      log_probabilities = row.log_probabilities.reshape(-1, num_output_symbols)
      probabilities = np.exp(log_probabilities)
      # np.exp(probabilities, out=probabilities)
      np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-3)
      # simple_decoder_output = []
      # for t in range(probabilities.shape[0]):
      #   best = np.argmax(probabilities[t,:])
      #   print(np.max(probabilities[t,:]))
      #   if (id_to_symbol[best] != "blank"):
      #     simple_decoder_output.append(id_to_symbol[best])

      # print("GALVEZ simple output:", "".join(simple_decoder_output))

      cutoff_prob = 1.0
      cutoff_top_n = 100
      scorer = ds_ctcdecoder.Scorer()
      result = scorer.init(scorer_path.encode('utf-8'), alphabet_path.encode('utf-8'))
      scorer.set_utf8_mode(False)
      assert result == 0, result
      alphabet = ds_ctcdecoder.Alphabet()
      result = alphabet.init(alphabet_path.encode('utf-8'))
      assert not scorer.is_utf8_mode()
      assert result == 0, result
      scorer = None
      outputs = ds_ctcdecoder.ctc_beam_search_decoder(
        probabilities, alphabet, 100,
        cutoff_prob, cutoff_top_n, scorer)
      print(f"GALVEZ:output={outputs[0][1]}")
      print(f"GALVEZ:length={probabilities.shape[0] * 30. / 1000.}")
      transcripts.append(outputs[0][1])
  
    return pd.DataFrame({"path": pd.Series(transcripts)})
  return generate_lm


# 
def load_audio_id_text_id_mapping(spark, input_catalogue_path: str):
  df = spark.read.format('json').schema(ARCHIVE_ORG_SCHEMA).load(input_catalogue_path)

  exploded_df = df.withColumn("exploded_files", F.explode(df.files))
  
  filtered_exploded_df = exploded_df.where(
    # When a file's size is 0 bytes, scripts/archive.org/download_items.py does
    # not download that file. We therefore filter out size 0 bytes to prevent
    # file-not-found errors in aling.py::load_transcripts()
    (exploded_df.exploded_files.size.cast(T.LongType()) != 0)
    &
    # This indicates that the file is not "private".
    # As far as I can tell, the "private" field is either "true" or null.
    # Trying to read this data as booleans turns every field null for some
    # reason, so it is currently a String field.
    # Private data is not downloadable by the general public.
    exploded_df.exploded_files.private.isNull()
    &
    # "[" and "]" are wild card characters. GCS has very poor support
    # for these. Namely, you can write them but not read them back. More
    # resources here: https://github.com/galv/lingvo-copy/issues/18
    # I simply filter out any files containing these characters for now.
    (~((exploded_df.identifier.contains("[")) |
       (exploded_df.identifier.contains("]"))))
    &
    (~((exploded_df.exploded_files["name"].contains("[")) |
       (exploded_df.exploded_files["name"].contains("]"))))
  )

  text_df = filtered_exploded_df.select(
    exploded_df.identifier,
    exploded_df.exploded_files["name"].alias("text_document_id"),
    exploded_df.exploded_files.format.alias("text_document_format")).where(
      (exploded_df.exploded_files.format == 'SubRip')
    )

  audio_df = filtered_exploded_df.select(
    exploded_df.identifier,
    exploded_df.exploded_files["name"].alias("audio_document_id")).where(
      ((exploded_df.exploded_files.format == 'MP3') |
       (exploded_df.exploded_files.format == 'VBR MP3'))
      &
      # Some non-mp3 files are given the format "MP3". See here:
      # https://ia802901.us.archive.org/4/items/disneychannelourhourafterhourafterhourprankstermarathonapril12004/disneychannelourhourafterhourafterhourprankstermarathonapril12004_files.xml
      (exploded_df.exploded_files["name"].endswith('.mp3'))
    )

  # The cause of the problem
  joined_df = audio_df.join(text_df, "identifier")

  return joined_df

def language_id():
  pass

