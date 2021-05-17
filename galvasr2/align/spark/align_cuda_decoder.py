"""
New forced alignment systemusing kaldi's cudadecoder implementation.
"""

from random import shuffle
import shlex
import subprocess
import logging
import os
from typing import List, Tuple

from absl import app
from absl import flags

import pyspark
import pyspark.sql.functions as F

from galvasr2.align.spark.align_lib import load_audio_id_text_id_mapping, load_audio_files

FLAGS = flags.FLAGS

flags.DEFINE_integer('stage',
                    1,
                    '')
flags.DEFINE_string('input_catalogue',
                    'gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz',
                    'Input catalogue. Basically just a dump of archive.org metadata for now.')
flags.DEFINE_string('input_dir',
                    '/root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS',
                    'Input directory. Exact format of this is a bit undefined right now and will likely change.')
flags.DEFINE_string('output_dir',
                    '/development/lingvo-source/output_ctm_dir3',
                    # '/root/the-peoples-speech-west-europe-bucket/forced-aligner/cudadecoder_ctm',
                    'Input directory. Exact format of this is a bit undefined right now and will likely change.')


def create_wav_scp(wav_scp_file_name: str, rows: List[pyspark.Row], base_path: str):
  with open(wav_scp_file_name, "w") as fh:
    lines = []
    for row in rows:
      key = os.path.join(row.identifier, row.audio_document_id)
      path = os.path.join(base_path, key)
      value = f"/usr/bin/sox \"{path}\" -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - |"
      line = f"{row.int64_uttid} {value}\n"
      lines.append(line)
      fh.write(line)
    # shuffle(lines)
    # fh.writelines(lines)

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

  wav_scp = "/development/lingvo-source/wav.scp"
  if FLAGS.stage <= 0:
    catalogue_df = load_audio_id_text_id_mapping(spark, FLAGS.input_catalogue)
    catalogue_df = catalogue_df.withColumn("int64_uttid", F.monotonically_increasing_id())
    catalogue_df = catalogue_df.cache()
    catalogue_df.write.mode("overwrite").format("csv").save("/development/lingvo-source/ids.csv")
    # paths_df = catalogue_df.select(F.concat(F.lit('/root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/'), F.col('identifier'), F.lit('/'), F.col('audio_document_id')))
    # paths_df.coalesce(1).write.mode("overwrite").format("text").save("/development/lingvo-source/paths.txt")
    training_sample_rows = catalogue_df.collect()
    catalogue_df.unpersist()

    create_wav_scp(wav_scp, training_sample_rows, FLAGS.input_dir)

  # /development/lingvo-source/output_ctm_dir/

  # nvprof --analysis-metrics -o  decoder-analysis.nvprof \
  # We want only the best path, so we set lattice-beam to 0.1
  # --main-q-capacity=35000 \
  # Can get 266x RTF with this configuration. Keep it?
  # bath size of 100 and num channels of 100 works just fine

  if FLAGS.stage <= 1:
    cmd = f"""
  /opt/kaldi/src/cudadecoderbin/batched-wav-nnet3-cuda3 \
  --frame-subsampling-factor=3 \
  --config=/opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/conf/online.conf \
  --max-active=7000 \
  --beam=15.0 \
  --lattice-beam=0.1 \
  --acoustic-scale=1.0 \
  --file-limit=1000 \
  --cuda-decoder-copy-threads=2 \
  --cuda-worker-threads={os.cpu_count()} \
  --segmentation=true \
  --cuda-use-tensor-cores=true \
  --max-batch-size=150 \
  --num-channels=250 \
  --lattice-postprocessor-rxfilename=/development/lingvo-source/lattice_postprocess.conf \
  --word-symbol-table=/opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/words.txt \
  /opt/kaldi/egs/aspire/s5/exp/chain/tdnn_7b/final.mdl \
  /opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/HCLG.fst \
  scp,p:{wav_scp} \
  {FLAGS.output_dir}
  """
    subprocess.check_call(shlex.split(cmd))

  # WARNING (batched-wav-nnet3-cuda3[5.5]:Read():feat/wave-reader.h:197) Exception caught in WaveHolder::Read(). kaldi::KaldiFatalError
  # WARNING (batched-wav-nnet3-cuda3[5.5]:EnsureObjectLoaded():util/kaldi-table-inl.h:317) Failed to load object from '/usr/bin/sox /root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/16628CC20161007COW/16628 CC-2016-1007-COW.mp3 -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - |'
  # WARNING (batched-wav-nnet3-cuda3[5.5]:Close():kaldi-io.cc:515) Pipe /usr/bin/sox /root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/16628CC20161007COW/16628 CC-2016-1007-COW.mp3 -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - | had nonzero return status 512
  
  # --verbose=10 \
  # subprocess.check_call(shlex.split(cmd))
  # audio_df = load_audio_files(spark, training_sample_rows, FLAGS.input_dir)
  # /opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/phones/word_boundary.int
  # /opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/phones/word_boundary.int
  # --word-boundary-rxfilename=/opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/phones/word_boundary.int \

if __name__ == '__main__':
  app.run(main)
