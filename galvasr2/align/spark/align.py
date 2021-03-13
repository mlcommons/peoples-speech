# bazel run galvasr2:spark_forced_aligner

from functools import partial
from io import BytesIO
import json
import logging
import os
import subprocess
import shlex
import tempfile
from typing import List
import wave

from ftfy import fix_text, guess_bytes
import langid
import numpy as np
import pandas as pd
import re
import srt

from absl import app
from absl import flags


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
import pyspark.sql.functions as F
from pyspark.sql.functions import array, array_contains, count, explode, lit
from pyspark.sql.types import ArrayType, BinaryType, DoubleType, FloatType, ShortType, StructType, StructField, StringType, IntegerType, LongType

# from lingvo.tools.audio_lib import DecodeToWav

from galvasr2.align.align import BEAM_WIDTH
from galvasr2.align.audio import AudioFormat, vad_split
from galvasr2.align.generate_lm import convert_and_filter_topk, build_lm
from galvasr2.align.generate_package import create_bundle
from galvasr2.align.spark.schemas import ARCHIVE_ORG_SCHEMA
from galvasr2.align.spark.event_listener import WriteTaskEndListener

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer('stage', 0, 'Stage of data pipeline to start from')
flags.DEFINE_string('work_dir',
                    'gs://the-peoples-speech-west-europe/forced-aligner/vad-segments-dump/Nov_6_2020/ALL_CAPTIONED_DATA_005',
                    'Directory under which intermediate and final outputs are dumped')
flags.DEFINE_string('input_dir',
                    'gs://the-peoples-speech-west-europe/archive_org/small_dataset',
                    'Input directory. Exact format of this is a bit undefined right now and will likely change.')
flags.DEFINE_string('input_catalogue',
                    'gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA.jsonl.gz',
                    'Input catalogue. Basically just a dump of archive.org metadata for now.')
flags.DEFINE_string('align_model_dir',
                    'gs://the-peoples-speech-west-europe/training_logs/galvez/tpu_ctc_5a',
                    'Directory holding lingvo acoustic model which will be used for alignment.')
flags.DEFINE_string('mozilla_ds_alphabet_txt',
                    '/development/lingvo-source/galvasr2/temporary_hardcoded_alphabet.txt',
                    'tokens.txt in Kaldi\'s format. This will be used to create Mozilla DeepSpeech\'s alphabet.txt')

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
def infer_language_func(text_column: pd.Series) -> pd.Series:
  return text_column.apply(lambda string: langid.classify(string)[0] if string else "")


# bytes, length, sampling_frequency, number_channels
def load_audio_files(spark, base_path: str):
    raw_audio_df = (spark.read.format("binaryFile")
                    .option("pathGlobFilter", "*.mp3")
                    .option("recursiveFileLookup", "true")
                    .load(base_path))
    
    return raw_audio_df.select('content',
                               F.reverse(F.split(raw_audio_df.path, "[.]"))[0].alias("format"),
                               # We will have repeats with this form of ID... It does not fulfill the purpose of an primary key...
                               # 44635        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/01-Ml.Z.Ragi-JinnandJadoo18.05.05.asr.srt
                               # 53884        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/02-Ml.Z.Ragi-JinnandJadoo25.05.05.asr.srt
                               # 55971        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/03-Ml.Z.Ragi-JinnandJadoo01.06.05.asr.srt
                               # 48287        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/04-Ml.Z.Ragi-JinnandJadoo08.06.05.asr.srt
                               # 44184        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/05-Ml.Z.Ragi-JinnandJadoo22.06.05.asr.srt
                               # 29040        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/06-Ml.Z.Ragi-JinnandJadoo29.06.05.asr.srt
                               # 53849        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/07-Ml.Z.Ragi-JinnandJadoo20.07.05.asr.srt
                               # 54745        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/08-Ml.Z.Ragi-JinnandJadoo27.07.05.asr.srt
                               # 44990        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/09-Ml.Z.Ragi-JinnandJadoo03.08.05.asr.srt
                               # 47756        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/10-Ml.Z.Ragi-JinnandJadoo10.08.05.asr.srt
                               # 46275        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/11-Ml.Z.Ragi-JinnandJadoo07.09.05.asr.srt
                               # 35660        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/12-Ml.Z.Ragi-JinnandJadoo14.09.05.asr.srt
                               # 50201        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/13-Ml.Z.Ragi-JinnandJadoo21.09.05.asr.srt
                               # I probably ought to use the non-format part of the final file path... That would work.
                               F.reverse(F.split(raw_audio_df.path, "/"))[1].alias("audio_document_id"),
                               F.monotonically_increasing_id().alias("int64_audio_document_id")
    )

@pandas_udf(StringType())
def fix_text_udf(binary_column: pd.Series) -> pd.Series:
  return binary_column.apply(lambda b: fix_text(guess_bytes(b)[0]))

# https://spark.apache.org/docs/3.0.2/api/python/pyspark.sql.html#pyspark.sql.HiveContext.newSession
def load_transcripts(spark, base_path: str, collected_text_document_rows: List[pyspark.Row]):
  text_document_ids = [os.path.join(base_path, row.identifier, row.text_document_id)
                       for row in collected_text_document_rows]
  text_document_ids = [path for path in text_document_ids
                       if "[" not in path and "]" not in path]
  # "[" and "]" are escape card characters. GCS has very poor support
  # for these. Namely, you can write them but not read them back. More
  # resources here: https://github.com/galv/lingvo-copy/issues/18
  # I simply filter out any files containing these characters for now.
  srt_df = (spark.read.format("binaryFile")
            .load(text_document_ids))
  # Note the duplication with load_audio_files
  return srt_df.select(srt_to_text(fix_text_udf(srt_df.content)).alias('transcript'),
                       F.reverse(F.split(srt_df.path, "/"))[1].alias("id"))

def prepare_vad_udf(num_padding_frames, threshold, aggressiveness, frame_duration_ms):
  # Each audio file returns multiple voiced fragments. I need an Array, don't I?
  return_type = StructType(
    [
      StructField("start_ms", ArrayType(IntegerType())),
      StructField("end_ms", ArrayType(IntegerType())),
      StructField("voiced_buffer", ArrayType(ArrayType(ShortType()))),
    ]
  )
  AUDIO_FORMAT = AudioFormat(sample_rate=16_000, channels=1, sample_byte_width=2)
  FRAME_DURATION_SAMPLES = (AUDIO_FORMAT.sample_rate * frame_duration_ms) // 1000
  FRAME_DURATION_BYTES = (FRAME_DURATION_SAMPLES * AUDIO_FORMAT.channels * 
                          AUDIO_FORMAT.sample_byte_width)
  @pandas_udf(return_type)
  def vad(audio_series: pd.Series, audio_types_series: pd.Series) -> pd.DataFrame:
    df_rows = []
    for audio_buffer, audio_type in zip(audio_series, audio_types_series):
      wav_bytes_buffer = BytesIO(DecodeToWavPipe(audio_buffer, audio_type))
      with wave.open(wav_bytes_buffer, "rb") as fh:
        num_frames = fh.getnframes()
        assert fh.getframerate() == AUDIO_FORMAT.sample_rate
        assert fh.getnchannels() == AUDIO_FORMAT.channels
        assert fh.getsampwidth() == AUDIO_FORMAT.sample_byte_width
        pcm_buffer = fh.readframes(num_frames)
        del wav_bytes_buffer
        num_frames = len(pcm_buffer) // FRAME_DURATION_BYTES
        # Can we lazily generate this? Yes.
        buffers = [pcm_buffer[FRAME_DURATION_BYTES * i: FRAME_DURATION_BYTES * (i + 1)] for i in range(num_frames)]
        del pcm_buffer
        generator = vad_split(buffers, AUDIO_FORMAT, num_padding_frames, 
                              threshold, aggressiveness)
        
        voiced_buffer_list, start_ms_list, end_ms_list = [], [], []
        total_serialized_bytes = 0
        for voiced_buffer, start_ms, end_ms in generator:
          # total_serialized_bytes += 2 * len(voiced_buffer)
          # if total_serialized_bytes > 2 * 1024 * 1024 * 1024 - 1024 * 1024 * 1024:
          #   print("WARNING: truncating voice-activity-detected audio to less than 2GB")
          #   break
          voiced_buffer_list.append(np.frombuffer(voiced_buffer, dtype=np.int16))
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

RESCORE_WITH_LM_OUTPUT_SCHEMA=StructType([StructField("transcribed_fragment", StringType())])
@pandas_udf(RESCORE_WITH_LM_OUTPUT_SCHEMA)
def rescore_with_lm(pdf: pd.DataFrame) -> pd.DataFrame:
  scorer = None
  for row in pdf.iterrows():
    # How else to create scorer?
    # scorer = Scorer(alpha=,
    #                 beta=,
    #                 scorer_path=row["scorer_path"],
    #                 alphabet=,)
    # scorer.load_lm()???
    for chunk_log_probabilities in row["log_probabilities"]:
      probabilities = chunk_log_probabilities
      # Do this in-place, overwriting the log_probabilities. We don't
      # need them.
      probabilities.exp(out=probabilities)
      BLANK_ID = 0
      # Mozilla DeepSpeech's ctc beam search decoder
      # DeepSpeech/native_client/ctcdecode/ctc_beam_search_decoder.cpp
      probabilities[:, [BLANK_ID -1]] = probabilities[:, [-1, BLANK_ID]]
      # Copy-pasta'd parameters from DeepSpeech/native_client/deepspeech.cc
      cutoff_prob = 1.0
      cutoff_top_n = 40
      ctc_beam_search_decoder(probabilities, alphabet, BEAM_WIDTH,
                              cutoff_prob, cutoff_top_n, scorer)
  return


GENERATE_LM_OUTPUT_SCHEMA=StringType()
def prepare_generate_lm_udf(kenlm_path: str, debug_work_dir: str, alphabet_path: str):
  
  @pandas_udf(GENERATE_LM_OUTPUT_SCHEMA)
  def generate_lm(transcript_series: pd.Series, text_document_id_series: pd.Series) -> pd.Series:
    scorer_paths = []
    for transcript, text_document_id in zip(transcript_series, text_document_id_series):
      # with tempfile.TemporaryDirectory(prefix=text_document_id, dir=debug_work_dir) as work_dir, \
      #      tempfile.NamedTemporaryFile('w+t', dir=work_dir) as input_txt:
      with tempfile.NamedTemporaryFile('w+t', dir=debug_work_dir) as input_txt:
        input_txt.write(transcript)
        input_txt.flush()
        scorer_path = os.path.join(debug_work_dir, text_document_id + ".scorer")
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

        scorer_paths.append(scorer_path)
      
    return pd.Series(scorer_paths)
  return generate_lm

def main(argv):
  spark = SparkSession.builder \
                      .master("local[1]") \
                      .appName("Forced Aligner") \
                      .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                      .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1")\
                      .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                      .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                      .config("spark.driver.memory", "7g")\
                      .config("spark.executor.memory", "7g")\
                      .config("spark.task.maxFailures", "2")\
                      .getOrCreate()
  spark.sparkContext.setLogLevel("INFO") # "ALL" for very verbose logging
  logging.getLogger("py4j").setLevel(logging.ERROR)
  pyspark.java_gateway.ensure_callback_server_started(spark.sparkContext._gateway)
  # spark.sparkContext._gateway.start_callback_server()
  listener = WriteTaskEndListener()
  spark.sparkContext._jsc.sc().addSparkListener(listener)

  vad_out_dir = os.path.join(FLAGS.work_dir, "vad_pcm_tfrecords")
  if FLAGS.stage <= 0:
    audio_df = load_audio_files(spark, FLAGS.input_dir)
    vad_udf = prepare_vad_udf(num_padding_frames=10, threshold=0.5,
                              aggressiveness=0, frame_duration_ms=30)
    vad_df = audio_df.withColumn("vad", vad_udf(audio_df.content, audio_df.format))
    vad_df = vad_df.withColumn("num_utterances_in_audio_document", F.size(vad_df.vad.voiced_buffer))

    exploded_voiced_buffer_df = vad_df.select(vad_df.audio_document_id,
                                              vad_df.int64_audio_document_id,
                                              vad_df.num_utterances_in_audio_document,
                                              F.posexplode(vad_df.vad.voiced_buffer))

    tfrecord_df = exploded_voiced_buffer_df.select(
      exploded_voiced_buffer_df.audio_document_id,
      exploded_voiced_buffer_df.int64_audio_document_id,
      exploded_voiced_buffer_df.col.alias("frames"),
      lit("-").alias("transcript"),
      F.concat_ws("-", exploded_voiced_buffer_df.audio_document_id, exploded_voiced_buffer_df.pos).alias("uttid"),
      F.monotonically_increasing_id().alias("int64_uttid"),
      exploded_voiced_buffer_df.num_utterances_in_audio_document,
    )

    tfrecord_df = tfrecord_df.withColumn("frames", F.expr("transform(frames, x -> float(x) * float(1./32768.))"))
    tfrecord_df.printSchema()

    tfrecord_df.write.mode("overwrite").format("tfrecord").option("recordType", "Example").save(vad_out_dir)

  if FLAGS.stage <= 1:
    # TODO: Compute this automatically
    # https://stackoverflow.com/questions/44082957/how-to-add-a-sparklistener-from-pyspark-in-python
    num_samples_written = listener.value
    if num_samples_written == 0:
      num_samples = spark.read.format("tfrecord").option("recordType", "Example").load(vad_out_dir).count()
    else:
      num_samples = num_samples_written

    # print(f"GALVEZ:num_samples_written={num_samples_written}")
    # print(f"GALVEZ:num_samples={num_samples}")
    # assert num_samples_written == num_samples



    # from IPython import embed; embed()
    # num_samples = 100_000
    # return

    # ctpu_up = subprocess.run(shlex.split("ctpu up -name forced-aligner-tpu -tpu-only -tpu-size v3-8 -tf-version 2.2"))

    TPU_IP = "10.240.1.2"

    # model_dir = "gs://the-peoples-speech-west-europe/PeoplesSpeech/ag_training/1127"
    model_dir = FLAGS.align_model_dir
    # model = "asr.inference_only.InferenceOnly"
    model = "asr.librispeech_ctc.TpuDecoderLibrispeech960Base"

    logits_dir = os.path.join(FLAGS.work_dir, "logits")

    def compute_max_steps(model_dir):
      # That the "train" directory is where the saved models are
      # stored is particular to lingvo. I don't expect this magic
      # constant to change.
      checkpoint_path = tf.train.latest_checkpoint(os.path.join(model_dir, "train"))
      step_pattern = r'-(\d+)$'
      checkpoint_step = int(re.search(step_pattern, checkpoint_path).group(1))
      max_steps = checkpoint_step + 1
      return max_steps

    #input.file_datasource.file_pattern:part-00000-8853e74a-fd03-46dc-affd-5c2ef87be96c-c000.tfrecord
    #part-00000-c4f0eb22-8f1e-45e2-9437-889428d09bf8-c000.tfrecord
    with tempfile.NamedTemporaryFile("w+") as fh:
      fh.write(f"""\
      input.file_datasource.file_pattern_prefix:{vad_out_dir}
      input.file_datasource.file_pattern:*.tfrecord
      input.num_samples:{num_samples}
      task.log_softmax_output_directory:{logits_dir}
      train.max_steps:{compute_max_steps(model_dir)}
      """)
      # This flush() is required. Otherwise, lingvo/trainer will see
      # an empty params file.
      fh.flush()

      # TODO: Make lingvo:trainer a dependency in the BUILD file. This is silly.
      subprocess.check_call(shlex.split(f"""
      lingvo/trainer --logdir={model_dir} \
      --model={model} \
      --logtostderr \
      --tpu=grpc://{TPU_IP}:8470 \
      --job=executor_tpu \
      --lingvo_executor_skip_saving_upon_stop \
      --model_params_file_override={fh.name}
      """))

  if FLAGS.stage <= 2:
    catalogue_df = spark.read.format('json').schema(ARCHIVE_ORG_SCHEMA).load(FLAGS.input_catalogue)
    load_transcripts(spark, FLAGS.input_dir, collected_text_document_rows)
    

    log_probabilities_schema = StructType([StructField("int64_uttid", IntegerType()),
                                           StructField("log_probabilities",
                                                       ArrayType(FloatType(), True))
    ])
    
    # log_probabilities_df = spark.read.format("tfrecord").schema(log_probabilities_schema).load(logits_dir)
    log_probabilities_df = spark.read.format("tfrecord").load(logits_dir)
    vad_df = spark.read.format("tfrecord").load(vad_out_dir)
    uttid_integer_mapping_df = vad_df.select(vad_df.int64_uttid, vad_df.uttid)
    log_probabilities_df = log_probabilities_df.join(uttid_integer_mapping_df, log_probabilities_df.int64_uttid == uttid_integer_mapping_df.int64_uttid, 'inner')
    log_probabilities_df = log_probabilities_df.drop(log_probabilities_df.int64_uttid)
    
    split_col = F.split(F.reverse(log_probabilities_df.uttid), '-', 2)
    log_probabilities_df = log_probabilities_df.withColumn('document_id', split_col.getItem(1))
    log_probabilities_df = log_probabilities_df.withColumn('utterance_id', split_col.getItem(0).cast(IntegerType()))
    log_probabilities_df = log_probabilities_df.groupBy('document_id').agg(collect_list("log_probabilities"), collect_list("utterance_id"))
    # TODO: Sort each array by utterance_id. array_sort lexicographically with a Struct?

    log_probabilities_df.join(text_df, col("log_probabilities_df.document_id") == col("transcript_df.document_id"), 'inner')

  if FLAGS.stage <= 3:
    generate_lm_udf = prepare_generate_lm_udf(
      "/install/kenlm/build/bin/",
      "/development/lingvo-source/tmpworkdir",
      FLAGS.mozilla_ds_alphabet_txt)
    df = spark.read.format("json").load("/home/ws15dgalvez/dumpblahblah.json")
    rows = df.select(generate_lm_udf(df.transcript, df.id)).head(1)
    from IPython import embed; embed()

if __name__ == '__main__':
  app.run(main)
