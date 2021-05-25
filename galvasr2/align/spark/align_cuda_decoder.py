"""
New forced alignment systemusing kaldi's cudadecoder implementation.
"""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import shlex
import subprocess
import logging
import os
import re
from typing import Dict, List, Tuple

from absl import app
from absl import flags
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

from galvasr2.align.spark.align_lib import fix_text_udf, load_audio_id_text_id_mapping, load_transcripts, prepare_create_audio_segments_udf, TemporaryMountDirectory
from galvasr2.align.spark.dsalign_lib import prepare_align_udf
import dsalign_main

FLAGS = flags.FLAGS

flags.DEFINE_integer('stage',
                    2,
                    '')
flags.DEFINE_string('input_catalogue',
                    'gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz',
                    'Input catalogue. Basically just a dump of archive.org metadata for now.')
flags.DEFINE_string('input_gcs_path',
                    'gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS',
                    'Input directory. Exact format of this is a bit undefined right now and will likely change.')
flags.DEFINE_string('input_gcs_bucket',
                    'gs://the-peoples-speech-west-europe',
                    '')
flags.DEFINE_string('input_dir',
                    '/root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS',
                    'Input directory. Exact format of this is a bit undefined right now and will likely change.')
flags.DEFINE_string('work_dir',
                    '/development/lingvo-source/output_work_dir_3h',
                    # '/root/the-peoples-speech-west-europe-bucket/forced-aligner/cudadecoder_ctm',
                    'Input directory. Exact format of this is a bit undefined right now and will likely change.')


def create_wav_scp(wav_scp_file_name: str, rows: List[pyspark.Row], base_path: str):
  with open(wav_scp_file_name, "w") as fh:
    lines = []
    for row in rows:
      key = os.path.join(row.identifier, row.audio_document_id)
      path = os.path.join(base_path, key)
      value = f"/usr/bin/sox \"{path}\" -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - |"
      line = f"{row.kaldi_normalized_uttid} {value}\n"
      lines.append(line)
      fh.write(line)
    # shuffle(lines)
    # fh.writelines(lines)

def split_wav_scp(posix_wav_scp, work_dir, num_splits):
  file_names = []
  file_handles = []
  split_dir = os.path.join(work_dir, "wav_scp_splits")
  os.makedirs(split_dir, exist_ok=True)
  for split in range(num_splits):
    file_names.append(os.path.join(split_dir, f"wav{split}.scp"))
    file_handles.append(open(file_names[-1], "w"))
  with open(posix_wav_scp) as fh:
    for i, line in enumerate(fh):
      i = i % 4
      file_handles[i].write(line)
  for fh in file_handles:
    fh.close()
  return file_names
  

def main(argv):
  spark = pyspark.sql.SparkSession.builder \
                        .master("local[*]")\
                        .config("spark.eventLog.enabled", "true")\
                        .config("spark.eventLog.dir", "/spark-events")\
                        .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                        .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                        .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                        .config("spark.history.fs.logDirectory", "/spark-events")\
                        .getOrCreate()
  spark.sparkContext.setLogLevel("INFO") # "ALL" for very verbose logging
  logging.getLogger("py4j").setLevel(logging.ERROR)

  catalogue_df = load_audio_id_text_id_mapping(spark, FLAGS.input_catalogue)
  # Kaldi's wav.scp format does not support space characters in the key field of a wav.scp file
  # We write the transcript to a file called "{kaldi_normalized_uttid}.ctm", so we also need to change all instances of "/" to "_"
  catalogue_df = catalogue_df.withColumn("kaldi_normalized_uttid", F.concat_ws("-", F.translate(catalogue_df.identifier, " /", "__"), F.translate(catalogue_df.audio_document_id, " /", "__")))
  # key_int_mapping = os.path.join(FLAGS.work_dir, "key_int_mapping_csv")
  if not FLAGS.work_dir.startswith("gs://"):
    os.makedirs(FLAGS.work_dir, exist_ok=True)
  wav_scp = os.path.join(FLAGS.work_dir, "wav.scp")
  if FLAGS.stage <= 0:
    catalogue_df = catalogue_df.cache()
    # catalogue_df.write.mode("overwrite").format("csv").options(header="true").save(key_int_mapping)
    training_sample_rows = catalogue_df.collect()
    catalogue_df.unpersist()

    with TemporaryMountDirectory(
            mount_cmd=["gcsfuse", "--implicit-dirs", FLAGS.input_gcs_bucket.lstrip("gs://")],
            unmount_cmd=["fusermount", "-u"]) as temp_dir_name:
      posix_wav_scp = re.sub(r'^{0}'.format(FLAGS.input_gcs_bucket),
                             temp_dir_name, wav_scp)
      create_wav_scp(posix_wav_scp, training_sample_rows, FLAGS.input_dir)

  # /development/lingvo-source/output_ctm_dir/

  # nvprof --analysis-metrics -o  decoder-analysis.nvprof \
  # We want only the best path, so we set lattice-beam to 0.1
  # --main-q-capacity=35000 \
  # Can get 266x RTF with this configuration. Keep it?
  # bath size of 100 and num channels of 100 works just fine

  ctm_out_dir = os.path.join(FLAGS.work_dir, "decoder_ctm_dir")
  if FLAGS.stage <= 1:
    if not FLAGS.work_dir.startswith("gs://"):
      os.makedirs(ctm_out_dir, exist_ok=True)
    with TemporaryMountDirectory(
            mount_cmd=["gcsfuse", "--implicit-dirs", FLAGS.input_gcs_bucket.lstrip("gs://")],
            unmount_cmd=["fusermount", "-u"]) as temp_dir_name:

      posix_ctm_out_dir = re.sub(r'^{0}'.format(FLAGS.input_gcs_bucket),
                                 temp_dir_name, ctm_out_dir)
      posix_wav_scp = re.sub(r'^{0}'.format(FLAGS.input_gcs_bucket),
                             temp_dir_name, wav_scp)
      posix_work_dir = re.sub(r'^{0}'.format(FLAGS.input_gcs_bucket),
                              temp_dir_name, FLAGS.work_dir)
      num_gpus = 4
      posix_wav_scp_shards = split_wav_scp(posix_wav_scp, posix_work_dir, num_gpus)

      executor = ThreadPoolExecutor(max_workers=num_gpus)
      def run_gpu(posix_wav_scp_shard, gpu_number):
        cmd = f"""\
  /opt/kaldi/src/cudadecoderbin/batched-wav-nnet3-cuda3 \
  --frame-subsampling-factor=3 \
  --config=/opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/conf/online.conf \
  --max-active=7000 \
  --beam=15.0 \
  --lattice-beam=0.1 \
  --acoustic-scale=1.0 \
  --cuda-decoder-copy-threads=2 \
  --cuda-worker-threads={os.cpu_count() // num_gpus} \
  --segmentation=true \
  --cuda-use-tensor-cores=true \
  --max-batch-size=150 \
  --num-channels=250 \
  --lattice-postprocessor-rxfilename=/development/lingvo-source/lattice_postprocess.conf \
  --word-symbol-table=/opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/words.txt \
  /opt/kaldi/egs/aspire/s5/exp/chain/tdnn_7b/final.mdl \
  /opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/HCLG.fst \
  scp,p:{posix_wav_scp_shard} \
  {posix_ctm_out_dir}
  """
        env = deepcopy(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = f"{gpu_number}"
        subprocess.check_call(shlex.split(cmd), env=env)
      for i, shard in enumerate(posix_wav_scp_shards):
        executor.submit(run_gpu, shard, i)
      executor.shutdown(wait=True)
          

  # 281 /usr/bin/sox "/root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/horror_express_ipod/horror_express.mp3" -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - |
  # 282 /usr/bin/sox "/root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/horror_express_ipod/horror_express.mp3" -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - |  
  # 281 /usr/bin/sox "/development/lingvo-source/horror_express.mp3" -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - |
  # 282 /usr/bin/sox "/development/lingvo-source/horror_express.mp3" -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - |
  
  # WARNING (batched-wav-nnet3-cuda3[5.5]:Read():feat/wave-reader.h:197) Exception caught in WaveHolder::Read(). kaldi::KaldiFatalError
  # WARNING (batched-wav-nnet3-cuda3[5.5]:EnsureObjectLoaded():util/kaldi-table-inl.h:317) Failed to load object from '/usr/bin/sox /root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/16628CC20161007COW/16628 CC-2016-1007-COW.mp3 -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - |'
  # WARNING (batched-wav-nnet3-cuda3[5.5]:Close():kaldi-io.cc:515) Pipe /usr/bin/sox /root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/16628CC20161007COW/16628 CC-2016-1007-COW.mp3 -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - | had nonzero return status 512
  
  # --verbose=10 \
  # subprocess.check_call(shlex.split(cmd))
  # audio_df = load_audio_files(spark, training_sample_rows, FLAGS.input_dir)
  # /opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/phones/word_boundary.int
  # /opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/phones/word_boundary.int
  # --word-boundary-rxfilename=/opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/phones/word_boundary.int \
  assert False
  if FLAGS.stage <= 2:
    FAKE_WORDS = {"<eps>", "<unk>", "[laughter]", "[noise]", "<s>", "</s>", "#0"}
    alphabet_set = set()
    with open("/opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/words.txt", "r") as fh:
      for line in fh:
        word = line.split(" ")[0]
        if word not in FAKE_WORDS:
          for character in word:
            alphabet_set.add(character)
      alphabet_set.add(" ")
    alphabet_path = "/development/lingvo-source/alphabet.txt"
    with open(alphabet_path, "w") as fh:
      for character in sorted(alphabet_set):
        fh.write(character)
        fh.write("\n")

    # TODO: Add options to DSAlign here
    dsalign_args = dsalign_main.parse_args("")

    align_udf = prepare_align_udf(dsalign_args, alphabet_path)

    ctm_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.ctm").load(ctm_out_dir)
    ctm_df = ctm_df.withColumn("kaldi_normalized_uttid", F.regexp_replace(F.reverse(F.split(ctm_df.path, "/"))[0], r"[.]ctm$", ""))
    ctm_df = ctm_df.withColumn("ctm_content", fix_text_udf(F.col("content"))).drop("path", "length", "modificationTime", "content")

    ctm_df = ctm_df.join(catalogue_df, "kaldi_normalized_uttid")
    downsampled_catalogue_df = ctm_df.drop("ctm_content")

    training_sample_rows = downsampled_catalogue_df.collect()
    transcripts_df = load_transcripts(spark, FLAGS.input_gcs_path, training_sample_rows)
    # TODO: Fix this. We need to recover the original identifier for each ctm file.
    ctm_df = ctm_df.join(transcripts_df, ['identifier', 'text_document_id'])

    # alignments_df = ctm_df.select(align_udf(F.concat(ctm_df.identifier, F.lit("/"), ctm_df.text_document_id),
    #                                         F.concat(ctm_df.identifier, F.lit("/"), ctm_df.audio_document_id),
    #                                         ctm_df.transcript, ctm_df.ctm_content))
    alignments_df = ctm_df.withColumn("alignments",
        align_udf(F.concat(ctm_df.identifier, F.lit("/"), ctm_df.text_document_id),
                  F.concat(ctm_df.identifier, F.lit("/"), ctm_df.audio_document_id),
                  ctm_df.transcript, ctm_df.ctm_content))
    print("GALVEZ:schema")
    alignments_df.printSchema()
    import sys; sys.stdout.flush()

    alignments_df.write.mode("overwrite").format("json").save(os.path.join(FLAGS.work_dir, "alignments_json"))

    pass
  training_data_export_dir = os.path.join(FLAGS.work_dir, "training_data_export")
  if FLAGS.stage <= 3:
    alignments_df = spark.read.json(os.path.join(FLAGS.work_dir, "alignments_json"))
    create_audio_segments_udf = prepare_create_audio_segments_udf(gs_bucket=FLAGS.input_gcs_bucket,
                                                                  output_dir=os.path.join(FLAGS.work_dir, "training_set")
    )
    audio_paths = F.concat(F.lit(FLAGS.input_gcs_path), F.lit("/"),
                           alignments_df.identifier, F.lit("/"),
                           alignments_df.audio_document_id)
    alignments_df = alignments_df.withColumn("output_paths", create_audio_segments_udf(
        audio_paths, alignments_df.identifier, alignments_df.audio_document_id,
        alignments_df.alignments.start_ms, alignments_df.alignments.end_ms))
    output_df = alignments_df.select(
        alignments_df.identifier,
        alignments_df.audio_document_id,
        alignments_df.text_document_id,
        F.struct(
            alignments_df.alignments.label,
            alignments_df.output_paths
        ).alias('training_data')
    )
    # coalesce(1) seems to make the create_audio_segments_udf function run serially
    output_df.write.mode("overwrite").json(os.path.join(FLAGS.work_dir, "dataset_manifest"))

if __name__ == '__main__':
  app.run(main)
