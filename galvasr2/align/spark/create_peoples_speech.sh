#!/bin/bash

set -euo pipefail

alignments_work_dir="gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b"
base_work_dir="gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/peoples-speech/"

# TODO: Filter out the test and dev sets Should affect only the CC-BY subsets.

echo "CC-BY Clean"

bazel run galvasr2/align/spark:cuda_decoder_forced_aligner -- --work_dir=gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b \
      --stage=3 \
      --max_cer=20.0 \
      --min_cer=0.0 \
      --max_duration_ms=16700 \
      --min_duration_ms=1000 \
      --license_filter="Not CC-BY-SA" \
      --alignments_work_dir=$alignments_work_dir \
      --work_dir=$base_work_dir/cc_by_clean \
      --number_of_shards=256

echo "CC-BY Dirty"

bazel run galvasr2/align/spark:cuda_decoder_forced_aligner -- --work_dir=gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b \
      --stage=3 \
      --max_cer=36.0 \
      --min_cer=20.0 \
      --max_duration_ms=16700 \
      --min_duration_ms=1000 \
      --license_filter="Not CC-BY-SA" \
      --alignments_work_dir=$alignments_work_dir \
      --work_dir=$base_work_dir/cc_by_dirty \
      --number_of_shards=256

echo "CC-BY-SA Clean"

bazel run galvasr2/align/spark:cuda_decoder_forced_aligner -- --work_dir=gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b \
      --stage=3 \
      --max_cer=20.0 \
      --min_cer=0.0 \
      --max_duration_ms=16700 \
      --min_duration_ms=1000 \
      --license_filter="CC-BY-SA" \
      --alignments_work_dir=$alignments_work_dir \
      --work_dir=$base_work_dir/cc_by_sa_clean \
      --number_of_shards=256

echo "CC-BY-SA Dirty"

bazel run galvasr2/align/spark:cuda_decoder_forced_aligner -- --work_dir=gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b \
      --stage=3 \
      --max_cer=36.0 \
      --min_cer=20.0 \
      --max_duration_ms=16700 \
      --min_duration_ms=1000 \
      --license_filter="CC-BY-SA" \
      --alignments_work_dir=$alignments_work_dir \
      --work_dir=$base_work_dir/cc_by_sa_dirty \
      --number_of_shards=256

