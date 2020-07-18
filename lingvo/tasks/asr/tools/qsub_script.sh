#!/bin/bash
#$ -cwd
#$ -j yes
#$ -l mem_free=5G,ram_free=5G,arch=*64*
#$ -v PATH

. ./librispeech_lib.sh

export CUDA_VISIBLE_DEVICES=""
source "/export/bdc01/home2/dgalvez/miniconda3/etc/profile.d/conda.sh"
conda activate galvasr

/export/b02/ws15dgalvez/galvASR2/third_party/lingvo/bazel-bin/lingvo/tools/create_asr_features --logtostderr --input_tarball=gs://the-peoples-speech-west-europe/Librispeech/raw/train-other-500.tar.gz --generate_tfrecords --shard_id=3 --num_shards=10 --num_output_shards=100 --output_range_begin=65 --output_range_end=70 --output_template=gs://the-peoples-speech-west-europe/Librispeech/train/train.tfrecords-%5.5d-of-%5.5d
