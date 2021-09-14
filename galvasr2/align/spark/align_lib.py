from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
import logging
import os
import pprint
import json
import shlex
import subprocess
import sys
import tempfile
import threading
from typing import List, Tuple
import wave

import ds_ctcdecoder
from ftfy import fix_text, guess_bytes
import langid
import numpy as np
import pandas as pd
import pydub
from pydub import AudioSegment
import pyspark
from matching.games import HospitalResident
import tensorflow as tf
import tqdm

import re
import srt

from pyspark.sql.functions import col, pandas_udf
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import array, array_contains, count, explode, lit
from pyspark.sql.types import (ArrayType, BooleanType, BinaryType, DoubleType,
                               FloatType, ShortType, StructType, StructField,
                               StringType, IntegerType, LongType)

from galvasr2.align.spark.schemas import ARCHIVE_ORG_SCHEMA
from galvasr2.align.spark.timeout import timeout

def DecodeToWavPipe(input_bytes, fmt):
  cmd = f'sox -t {fmt} - -t wav --channels 1 --rate 16000 --encoding signed --bits 16 -'
  p = subprocess.Popen(shlex.split(cmd),
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
  out, err = p.communicate(input=input_bytes)
  assert p.returncode == 0, err
  return out

def DecodeToRawPipe(input_bytes, fmt):
  cmd = f'sox -t {fmt} - -t raw --channels 1 --rate 16000 --encoding signed --bits 16 -'
  p = subprocess.Popen(shlex.split(cmd),
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
  out, err = p.communicate(input=input_bytes)
  assert p.returncode == 0, err
  return out


def EncodeFromRawPipe(input_bytes, fmt):
  cmd = f'sox -t raw --channels 1 --rate 16000 --encoding signed --bits 16 - -t {fmt} -'
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
      print("WARNING: trouble parsing srt file content")
      print(exc)
      return ""
  return srt_file_contents.apply(helper)

@pandas_udf(StringType())
def normalize_english_text_udf(unnormalized_text_series: pd.Series) -> pd.Series:
    from gruut.lang import get_tokenizer

    tokenizer = get_tokenizer(
        "en",
        no_pos=True,
        # use_number_converters=args.number_converters,
        do_replace_currency=True,
        exclude_non_words=False,
    )
    normalized_texts = []
    for unnormalized_text in unnormalized_text_series:
        normalized_text = " ".join(str(sentence.clean_text) for sentence in
                                   tokenizer.tokenize(unnormalized_text))
        normalized_texts.append(normalized_text)
    return pd.Series(normalized_texts)


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
                             F.reverse(F.split(raw_audio_df.path, "/"))[1].alias("identifier"),
                             F.monotonically_increasing_id().alias("int64_audio_document_id")
  )

# (100k-hours-lingvo-3) root@3e330ec805fd:/development/lingvo-source# gsutil ls gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/house.hbs.mars.hrs05H_A1310_100511/
# gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/house.hbs.mars.hrs05H_A1310_100511/hrs05H_A1310_100511.asr.srt
# gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/house.hbs.mars.hrs05H_A1310_100511/hrs05H_A1310_100511.mp3
# pyspark.sql.utils.AnalysisException: Path does not exist: gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/gov.house.hbs.hrs05H_A1310_100511/hrs05H_A1310_100511.asr.srt

@pandas_udf(StringType())
def fix_text_udf(binary_column: pd.Series) -> pd.Series:
  return binary_column.apply(lambda b: fix_text(guess_bytes(b)[0]))

# https://spark.apache.org/docs/3.0.2/api/python/pyspark.sql.html#pyspark.sql.HiveContext.newSession
def load_transcripts(spark, base_path: str, collected_text_document_rows: List[pyspark.Row]):
  def fix_name(identifier, text_document_id):
    if (identifier == "gov.house.hbs.hrs05H_A1310_100511" and
        text_document_id == "hrs05H_A1310_100511.asr.srt"):
      return "hrs05H_A1310_100511.auto.srt"
    else:
      return text_document_id
  # TODO: Upload this file
  with open("/development/lingvo-source/missing_files.json", "r") as fh:
    missing_text_document_ids = set(json.load(fh))
  text_document_ids = [os.path.join(base_path, row.identifier, fix_name(row.identifier, row.text_document_id))
                       for row in collected_text_document_rows]
  text_document_ids = [tid for tid in text_document_ids if tid not in missing_text_document_ids]
  srt_df = (spark.read.format("binaryFile")
            .load(text_document_ids))
  # Note the duplication with load_audio_files
  return srt_df.select(srt_to_text(fix_text_udf(srt_df.content)).alias('transcript'),
                       F.reverse(F.split(srt_df.path, "/"))[0].alias("text_document_id"),
                       F.reverse(F.split(srt_df.path, "/"))[1].alias("identifier"),
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

def load_audio_and_text_dfs(spark, input_catalogue_path: str):
  df = spark.read.format('json').schema(ARCHIVE_ORG_SCHEMA).load(input_catalogue_path)

  exploded_df = df.withColumn("exploded_files", F.explode(df.files))
  
  filtered_exploded_df = exploded_df.where(
    # When a file's size is 0 bytes, scripts/archive.org/download_items.py does
    # not download that file. We therefore filter out size 0 bytes to prevent
    # file-not-found errors in align_lib.py::load_transcripts()
    # https://archive.org/metadata/house.hbs.mars.hrs06RES2154_090401
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

  # from IPython import embed; embed()

  print("GALVEZ:json=", df.count())
  print("GALVEZ:exploded=", filtered_exploded_df.count())

  text_df = filtered_exploded_df.select(
    filtered_exploded_df.identifier,
    filtered_exploded_df.exploded_files["name"].alias("text_document_id"),
    filtered_exploded_df.exploded_files.format.alias("text_document_format")).where(
      (filtered_exploded_df.exploded_files.format == 'SubRip')
    )

  audio_df = filtered_exploded_df.select(
    filtered_exploded_df.identifier,
    filtered_exploded_df.exploded_files["name"].alias("audio_document_id")).where(
      ((filtered_exploded_df.exploded_files.format == 'MP3') |
       (filtered_exploded_df.exploded_files.format == 'VBR MP3'))
      &
      # Some non-mp3 files are given the format "MP3". See here:
      # https://ia802901.us.archive.org/4/items/disneychannelourhourafterhourafterhourprankstermarathonapril12004/disneychannelourhourafterhourafterhourprankstermarathonapril12004_files.xml
      (filtered_exploded_df.exploded_files["name"].endswith('.mp3'))
    )
  return audio_df, text_df

# Three columns:
# identifer, MP3 file name, transcript file name
def load_audio_id_text_id_mapping(spark, input_catalogue_path: str):
  audio_df, text_df = load_audio_and_text_dfs(spark, input_catalogue_path)

  joined_df = audio_df.join(text_df, "identifier")
  joined_df = joined_df.withColumn("levenshtein", F.levenshtein(joined_df.audio_document_id, joined_df.text_document_id))
  audio_to_text_mapping_df = joined_df.groupBy("identifier").applyInPandas(fuzzy_matching, schema=FUZZY_MATCHING_RETURN_TYPE)
  return audio_to_text_mapping_df

@pandas_udf(BooleanType())
def file_exists_udf(path_series: pd.Series) -> pd.Series:
    # print("GALVEZ:", path_series[0])
    # print("GALVEZ:", len(path_series))
    # import sys; sys.stdout.flush()
    with ThreadPoolExecutor(80) as executor:
        return pd.Series(list(tqdm.tqdm(executor.map(tf.io.gfile.exists, path_series), total=len(path_series))))

FUZZY_MATCHING_RETURN_TYPE = StructType([StructField("identifier", StringType()),
                                         StructField("audio_document_id", StringType()),
                                         StructField("text_document_id", StringType()),
])
def fuzzy_matching(identifier: Tuple, pdf: pd.DataFrame):
  # Default recusion limit of 1000 causes the deepcopy in
  # HospitalResident.create_from_dictionaries to fail for particularly
  # large inputs.
  (identifier, ) = identifier
  identifier = str(identifier) # instead of np.str
  sys.setrecursionlimit(10000)
  audio_preferences = defaultdict(list)
  text_preferences = defaultdict(list)
  for row in pdf.itertuples():
    audio_preferences[row.audio_document_id].append((row.levenshtein,
                                                     row.text_document_id))
    text_preferences[row.text_document_id].append((row.levenshtein,
                                                   row.audio_document_id))
  # tuples are sorted lexicographically, so by levenshtein distance first
  for k, v in audio_preferences.items():
    audio_preferences[k] = [text_id for _score, text_id in sorted(v)]
  for k, v in text_preferences.items():
    text_preferences[k] = [audio_id for _score, audio_id in sorted(v)]
  # problem = StableMarriage.create_from_dictionaries(audio_preferences, text_preferences)

  more_audio_files = len(audio_preferences) > len(text_preferences)
  if more_audio_files:
    resident_preferences = audio_preferences
    hospital_preferences = text_preferences
  else:
    resident_preferences = text_preferences
    hospital_preferences = audio_preferences
  hospital_capacities = {k: 1 for k in hospital_preferences.keys()}


  try:
    problem = HospitalResident.create_from_dictionaries(resident_preferences,
                                                        hospital_preferences,
                                                        hospital_capacities)
  except RecursionError:
    print("GALVEZ:problem=", identifier[0])
    pprint.pprint("GALVEZ:residents:")
    pprint.pprint(resident_preferences)
    pprint.pprint("GALVEZ:hospitals:")
    pprint.pprint(hospital_preferences)
    raise
  pairs = problem.solve(optimal="hospital")

  pdf_dict = {"identifier": [], "audio_document_id": [], "text_document_id": []}
  for hospital_id, (resident_id, ) in pairs.items():
    if more_audio_files:
      audio_id = resident_id
      text_id = hospital_id
    else:
      audio_id = hospital_id
      text_id = resident_id
    pdf_dict["identifier"].append(identifier)
    pdf_dict["audio_document_id"].append(audio_id.name)
    pdf_dict["text_document_id"].append(text_id.name)

  return pd.DataFrame(pdf_dict)

class TemporaryMountDirectory(tempfile.TemporaryDirectory):
  # pylint: disable=no-member
  def __init__(self, mount_cmd: List, unmount_cmd: List):
    super().__init__()
    self._unmount_cmd = unmount_cmd
    subprocess.check_call(mount_cmd + [self.name])
    
  def __exit__(self, exc, value, tb):
    try:
      subprocess.check_call(self._unmount_cmd + [self.name])
      unmounted = True
    except subprocess.CalledProcessError:
      try:
        subprocess.check_call(["umount", self.name])
        unmounted = True
      except subprocess.CalledProcessError:
        unmounted = False
        print(f"WARNING: Failed to unmount {self.name}. Not removing temporary directory. Trying to remove a temporary directory that fails to unmount results in: OSError: [Errno 5] Input/output error")
    if unmounted:
      super().__exit__(exc, value, tb)

def _prepare_soxi_udf(soxi_flags, spark_return_type, python_return_type):
  @pandas_udf(spark_return_type)
  def get_soxi_info_udf(audio_file_series: pd.Series) -> pd.Series:
    # soxi does not support reading from stdin, so we need to use
    # gcsfuse-mounted posix paths
    durations = []
    with TemporaryMountDirectory(
            mount_cmd=["gcsfuse", "--implicit-dirs", "gs://the-peoples-speech-west-europe".lstrip("gs://")],
            unmount_cmd=["fusermount", "-u"]) as temp_dir_name:
      for audio_file in audio_file_series:
        audio_file = re.sub(r'^{0}'.format("gs://the-peoples-speech-west-europe"), temp_dir_name, audio_file)
        assert not audio_file.startswith("gs://")
        cmd = f"soxi {soxi_flags} \"{audio_file}\""
        try:
          duration = subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL, timeout=10) # 10 second timeout
          # print("GALVEZ:value=")
          # print(duration)
          duration = python_return_type(duration.rstrip(b'\n'))
        except subprocess.CalledProcessError:
          # WARNING: Assumes that your return type default constructor returns a "reasonable" value.
          # May return None instead?
          duration = python_return_type()
        except subprocess.TimeoutExpired:
          print(f"Restarting on {audio_file}")
          # Call again. Sometimes gcsfuse just stalls, so we need restartability
          return get_soxi_info_udf(audio_file_series)
        durations.append(duration)
      return pd.Series(durations)
  return get_soxi_info_udf

get_audio_seconds_udf = _prepare_soxi_udf("-D", DoubleType(), float)
get_audio_sample_rate_udf = _prepare_soxi_udf("-r", StringType(), str)
get_audio_annotations_udf = _prepare_soxi_udf("-a", BinaryType(), bytes)

def create_audio_segments_udf(audio_bytes_series: pd.Series, audio_type_series: pd.Series,
                              audio_names_series: pd.Series, start_ms_array_series: pd.Series,
                              end_ms_array_series: pd.Series) -> pd.DataFrame:
    for audio_bytes, audio_type, audio_name, start_ms_array, end_ms_array in zip(
            audio_bytes_series, audio_type_series, audio_name_series,
            start_ms_array_series, end_ms_array_series):
        assert audio_type == 'mp3'
        decoded_bytes = DecodeToRawPipe(audio_bytes, audio_type)
        audio_segment = pydub.AudioSegment(
            decoded_bytes, 
            frame_rate=16_000,
            sample_width=2,
            channels=1
        )
        for i, (start_ms, end_ms) in enumerate(zip(start_ms_array, end_ms_array)):        
            segment_flac_bytes = EncodeFromRawPipe(audio_segment[start_ms:end_ms].raw_data, "flac")
        pass
    pass

def prepare_filter_alignments_udf(cer_threshold: float, duration_ms_threshold: int):
    RETURN_TYPE = T.StructType([T.StructField("start_ms", T.ArrayType(T.LongType())),
                                T.StructField("end_ms", T.ArrayType(T.LongType())),
                                T.StructField("label", T.ArrayType(T.StringType())),
                                T.StructField("cer", T.ArrayType(T.FloatType())),
                                T.StructField("wer", T.ArrayType(T.FloatType())),
    ])
    @F.pandas_udf(RETURN_TYPE)
    def filter_alignments_udf(start_ms_arrays: pd.Series, end_ms_arrays: pd.Series,
                              label_arrays: pd.Series, cer_arrays: pd.Series,
                              wer_arrays: pd.Series) -> pd.DataFrame:
        for start_ms_array, end_ms_array, label_array, cer_array, cer_array in zip(start_ms_arrays, end_ms_arrays, label_arrays, cer_arrays, wer_arrays):
            pass

def prepare_create_audio_segments_udf(gs_bucket: str, output_dir: str):
  RETURN_TYPE = ArrayType(StringType())
  @pandas_udf(RETURN_TYPE)
  def create_audio_segments_udf(audio_file_gcs_paths: pd.Series, identifier_series: pd.Series,
                                audio_document_id_series: pd.Series, start_ms_arrays: pd.Series,
                                end_ms_arrays: pd.Series) -> pd.Series:
    chunk_paths = []
    with TemporaryMountDirectory(
            mount_cmd=["gcsfuse", "--implicit-dirs", gs_bucket.lstrip("gs://")],
            unmount_cmd=["fusermount", "-u"]) as temp_dir_name:
      if not output_dir.startswith("gs://"):
        posix_output_dir = output_dir
      else:
        posix_output_dir = re.sub(r'^{0}'.format(gs_bucket), temp_dir_name, output_dir)
      for audio_file_gcs_path, identifier, audio_document_id, start_ms_array, end_ms_array in zip(audio_file_gcs_paths, identifier_series, audio_document_id_series, start_ms_arrays, end_ms_arrays):
        chunk_paths.append([])
        audio_file_path = re.sub(r'^{0}'.format(gs_bucket), temp_dir_name, audio_file_gcs_path)
        print(f"GALVEZ:audio_file_path={audio_file_path}")
        try:
          source = AudioSegment.from_file(audio_file_path, subprocess_timeout=100)
        except subprocess.TimeoutExpired:
          print("GALVEZ:timed out, need to retry")
          return create_audio_segments_udf.func(audio_file_gcs_paths,
                                                identifier_series,
                                                audio_document_id_series,
                                                start_ms_arrays,
                                                end_ms_arrays)
        except pydub.exceptions.CouldntDecodeError:
          print(f"GALVEZ:problematic audio_file_path={audio_file_path}")
          continue
        identifier = identifier.replace('/', '_')
        this_file_output_dir = os.path.join(posix_output_dir, identifier)
        os.makedirs(this_file_output_dir, exist_ok=True)
        base, _ = os.path.splitext(audio_document_id)
        # We have to handle cases where audio_document_id contains a slash, like this one: collateral/gov.house.oversight.2007.03.19.iphone.mp3
        # We could alternatively, but I worry about users of the dataset naively writing a glob pattern like "*/*.mp3", rather than "*/*/*.mp3".
        # Furthermore there could be an arbitrary number of "/" characters. That is hard to handle programatically
        base = base.replace('/', '_')
        last_write_file_name = f"{base}-{(len(start_ms_array) - 1):04d}.wav"
        already_done = timeout(os.path.exists, (os.path.join(this_file_output_dir, last_write_file_name),), timeout_duration=100)
        if already_done:
          pass
        for i, (start_ms, end_ms) in enumerate(zip(start_ms_array, end_ms_array)):
          # Flac encoding probably good
          write_file_name = f"{base}-{i:04d}.flac"
          if not already_done:
            write_path = os.path.join(this_file_output_dir, write_file_name)
            try:
              fh = source[start_ms:end_ms].set_frame_rate(16_000).set_sample_width(2).set_channels(1).export(write_path, format="flac", subprocess_timeout=100)
            except subprocess.TimeoutExpired:
              print("GALVEZ:timed out 2, need to retry")
              return create_audio_segments_udf.func(audio_file_gcs_paths,
                                                    identifier_series,
                                                    audio_document_id_series,
                                                    start_ms_arrays,
                                                    end_ms_arrays)
            except pydub.exceptions.CouldntEncodeError:
              print(f"GALVEZ: Couldn't encode {write_path} [{start_ms}:{end_ms}]")
              continue
            else:
              fh.close()
          chunk_paths[-1].append(os.path.join(identifier, write_file_name))
      return pd.Series(chunk_paths)
  return create_audio_segments_udf

# (1, 300, Counter(),
#  [{'start': 1513510,
#    'end': 1529030,
#    'transcript': "[noise] but it's actually <unk> i believe what i remember is that we get the mayor to send a letter two said commissioner stating such and such and such we need a response from the commissioner um saving",
#    'text-start': 17857,
#    'text-end': 18030,
#    'meta': {},
#    'aligned-raw': 'I believe what I remember is that we get the mayor to send a letter to said Commissioner stating such and such and such we need a response from the commissioner. For saving ',
#    'aligned': 'ibelievewhatirememberisthatwegetthemayortosendalettertosaidcommissionerstatingsuchandsuchandsuchweneedaresponsefromthecommissioner.forsaving'}])


