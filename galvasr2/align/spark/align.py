import logging
import os
import subprocess
import shlex
import tempfile

from absl import app
from absl import flags


import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# from lingvo.tools.audio_lib import DecodeToWav

from galvasr2.align.align import align
from galvasr2.align.audio import AudioFormat, vad_split
from galvasr2.align.generate_lm import convert_and_filter_topk, build_lm
from galvasr2.align.generate_package import create_bundle
from galvasr2.align.spark.schemas import ARCHIVE_ORG_SCHEMA
from galvasr2.align.spark.event_listener import WriteTaskEndListener

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer('stage', 0, 'Stage of data pipeline to start from')
flags.DEFINE_string('work_dir',
                    'gs://the-peoples-speech-west-europe/forced-aligner/vad-segments-dump/Mar_13_2021/small_dataset_better_model_fixed_wav_scaling',
                    'Directory under which intermediate and final outputs are dumped')
flags.DEFINE_string('input_dir',
                    'gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA',
                    'Input directory. Exact format of this is a bit undefined right now and will likely change.')
flags.DEFINE_string('input_catalogue',
                    'gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA.jsonl.gz',
                    'Input catalogue. Basically just a dump of archive.org metadata for now.')
flags.DEFINE_string('align_model_dir',
                    'gs://the-peoples-speech-west-europe/training_logs/galvez/tpu_ctc_6h',
                    'Directory holding lingvo acoustic model which will be used for alignment.')
flags.DEFINE_string('mozilla_ds_alphabet_txt',
                    '/development/lingvo-source/galvasr2/temporary_hardcoded_alphabet.txt',
                    'tokens.txt in Kaldi\'s format. This will be used to create Mozilla DeepSpeech\'s alphabet.txt')

def main(argv):
  spark = SparkSession.builder \
                      .master("local[15]") \
                      .appName("Forced Aligner") \
                      .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                      .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1")\
                      .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                      .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                      .config("spark.driver.memory", "120g")\
                      .config("spark.executor.memory", "120g")\
                      .config("spark.task.maxFailures", "2")\
                      .config("spark.rpc.askTimeout", "480s")\
                      .config("spark.executor.heartbeatInterval", "20000ms")\
                      .config("spark.eventLog.enabled", "true")\
                      .config("spark.eventLog.dir", "/development/lingvo-source/spark-events")\
                      .config("spark.history.fs.logDirectory", "/development/lingvo-source/spark-events")\
                      .getOrCreate()
  spark.sparkContext.setLogLevel("INFO") # "ALL" for very verbose logging
  logging.getLogger("py4j").setLevel(logging.ERROR)
  pyspark.java_gateway.ensure_callback_server_started(spark.sparkContext._gateway)
  spark.sparkContext._gateway.start_callback_server()
  listener = WriteTaskEndListener()
  spark.sparkContext._jsc.sc().addSparkListener(listener)

  catalogue_df = load_audio_id_text_id_mapping(spark, FLAGS.input_catalogue)
  training_sample_rows = catalogue_df.collect()

  vad_out_dir = os.path.join(FLAGS.work_dir, "vad_pcm_tfrecords")
  audio_document_id_int64_id_dir = os.path.join(FLAGS.work_dir, "audio_document_id_to_int64_id")
  if FLAGS.stage <= 0:
    audio_df = load_audio_files(spark, training_sample_rows, FLAGS.input_dir)
    vad_udf = prepare_vad_udf(num_padding_frames=10, threshold=0.5,
                              aggressiveness=0, frame_duration_ms=30)
    vad_df = audio_df.withColumn("vad", vad_udf(audio_df.content,
                                                audio_df.format,
                                                audio_df.audio_document_id))
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

    tfrecord_df.write.mode("overwrite").format("tfrecord").option("recordType", "Example").save(vad_out_dir)

    vad_df.select(vad_df.audio_document_id, vad_df.int64_audio_document_id).write.mode("overwrite").format("json").save(audio_document_id_int64_id_dir)

  logits_dir = os.path.join(FLAGS.work_dir, "logits")
  if FLAGS.stage <= 1:
    num_samples_written = 11_175_308  # listener.value
    if num_samples_written == 0:
      # We are restarting immediately from stage 1, rather than
      # continuing from stage 0, so we need to recompute the number of
      # samples
      num_samples = spark.read.format("tfrecord").option("recordType", "Example").load(vad_out_dir).count()
    else:
      num_samples = num_samples_written

    TPU_IP = "10.240.1.2"

    model_dir = FLAGS.align_model_dir
    model = "asr.librispeech_ctc.TpuDecoderGrphm_DO_SpecAug_StackingSubSampler"

    def compute_max_steps(model_dir):
      # That the "train" directory is where the saved models are
      # stored is particular to lingvo. I don't expect this magic
      # constant to change.
      checkpoint_path = tf.train.latest_checkpoint(os.path.join(model_dir, "train"))
      step_pattern = r'-(\d+)$'
      checkpoint_step = int(re.search(step_pattern, checkpoint_path).group(1))
      max_steps = checkpoint_step + 1
      return max_steps

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

      cmd = f"""
      lingvo/trainer --logdir={model_dir} \
      --model={model} \
      --logtostderr \
      --tpu=grpc://{TPU_IP}:8470 \
      --job=executor_tpu \
      --lingvo_executor_skip_saving_upon_stop \
      --model_params_file_override={fh.name} \
      --ctc_inference_model_num_samples={num_samples} \
      --ctc_inference_model_batch_size=48
      """
      subprocess.check_call(shlex.split(cmd))

  assert False, "DSAlign won't work in its current state..."

  if FLAGS.stage <= 2:
    transcripts_df = load_transcripts(spark, FLAGS.input_dir, training_sample_rows)

    log_probabilities_schema = StructType([StructField("int64_uttid", LongType()),
                                           StructField("int64_audio_document_id", LongType()),
                                           StructField("log_probabilities",
                                                       ArrayType(FloatType(), True)),
                                           StructField("transcripts", StringType()),
    ])
    log_probabilities_df = spark.read.format("tfrecord").schema(log_probabilities_schema).load(logits_dir)
    # log_probabilities_df = log_probabilities_df.groupBy(log_probabilities_df.int64_audio_document_id).

    audio_document_integer_mapping_df = spark.read.format("json").load(audio_document_id_int64_id_dir)
    log_probabilities_df = log_probabilities_df.join(audio_document_integer_mapping_df,
                                                     "int64_audio_document_id")
    generate_lm_udf = prepare_generate_lm_udf(
      "/install/kenlm/build/bin/",
      "/development/lingvo-source/tmpworkdir",
      FLAGS.mozilla_ds_alphabet_txt)
    df = transcripts_df.join(catalogue_df, ['text_document_id']).join(log_probabilities_df, ['audio_document_id']).drop('int64_audio_document_id', 'text_document_format')
    # stuff = df.groupBy(df.identifier, df.text_document_id).applyInPandas(generate_lm_udf, GENERATE_LM_OUTPUT_SCHEMA)

  #   # log_probabilities_df = spark.read.format("tfrecord").load(logits_dir)
  #   uttid_integer_mapping_df = spark.read.format("json").load(audio_document_id_int64_id_dir)
  #   uttid_integer_mapping_df = vad_df.select(vad_df.int64_uttid, vad_df.uttid)
  #   log_probabilities_df = log_probabilities_df.join(uttid_integer_mapping_df, log_probabilities_df.int64_uttid == uttid_integer_mapping_df.int64_uttid, 'inner')
  #   log_probabilities_df = log_probabilities_df.drop(log_probabilities_df.int64_uttid)
    
  #   split_col = F.split(F.reverse(log_probabilities_df.uttid), '-', 2)
  #   log_probabilities_df = log_probabilities_df.withColumn('document_id', split_col.getItem(1))
  #   log_probabilities_df = log_probabilities_df.withColumn('utterance_id', split_col.getItem(0).cast(IntegerType()))
  #   log_probabilities_df = log_probabilities_df.groupBy('document_id').agg(collect_list("log_probabilities"), collect_list("utterance_id"))
  #   # TODO: Sort each array by utterance_id. array_sort lexicographically with a Struct?

  #   log_probabilities_df.join(text_df, col("log_probabilities_df.document_id") == col("transcript_df.document_id"), 'inner')

  # if FLAGS.stage <= 3:
  #   generate_lm_udf = prepare_generate_lm_udf(
  #     "/install/kenlm/build/bin/",
  #     "/development/lingvo-source/tmpworkdir",
  #     FLAGS.mozilla_ds_alphabet_txt)
  #   df = spark.read.format("json").load("/home/ws15dgalvez/dumpblahblah.json")
  #   rows = df.select(generate_lm_udf(df.transcript, df.id)).head(1)
  #   from IPython import embed; embed()

if __name__ == '__main__':
  app.run(main)
