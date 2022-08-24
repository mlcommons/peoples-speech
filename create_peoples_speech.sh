#!/bin/bash


./create_swap_disk.sh

set -euo pipefail

alignments_work_dir="gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b"
base_work_dir="gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/peoples-speech/output_work_dir_9a"

echo "CC-BY Clean"

# python galvasr2/align/spark/align_cuda_decoder.py \
#       --stage=3 \
#       --max_cer=10.0 \
#       --min_cer=0.0 \
#       --max_duration_ms=16700 \
#       --min_duration_ms=1000 \
#       --license_filter="Not CC-BY-SA" \
#       --alignments_work_dir=$alignments_work_dir \
#       --work_dir=$base_work_dir/cc_by_clean

echo "CC-BY Dirty"

python galvasr2/align/spark/align_cuda_decoder.py \
      --stage=3 \
      --max_cer=36.0 \
      --min_cer=10.0 \
      --max_duration_ms=16700 \
      --min_duration_ms=1000 \
      --license_filter="Not CC-BY-SA" \
      --alignments_work_dir=$alignments_work_dir \
      --work_dir=$base_work_dir/cc_by_dirty

echo "CC-BY-SA Clean"

python galvasr2/align/spark/align_cuda_decoder.py \
      --stage=3 \
      --max_cer=10.0 \
      --min_cer=0.0 \
      --max_duration_ms=16700 \
      --min_duration_ms=1000 \
      --license_filter="CC-BY-SA" \
      --alignments_work_dir=$alignments_work_dir \
      --work_dir=$base_work_dir/cc_by_sa_clean
echo "CC-BY-SA Dirty"

python galvasr2/align/spark/align_cuda_decoder.py \
      --stage=3 \
      --max_cer=36.0 \
      --min_cer=10.0 \
      --max_duration_ms=16700 \
      --min_duration_ms=1000 \
      --license_filter="CC-BY-SA" \
      --alignments_work_dir=$alignments_work_dir \
      --work_dir=$base_work_dir/cc_by_sa_dirty
