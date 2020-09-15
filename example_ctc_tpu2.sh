export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

bazel-bin/lingvo/trainer --logdir=$1 \
                         --mode=sync \
                         --model=asr.librispeech_ctc.Librispeech960Base5e4 \
                         --logtostderr \
                         --tpu=grpc://10.255.248.234:8470

# asr.librispeech_ctc.Librispeech960Base -> lingvo.tasks.asr.params.librispeech_ctc.Librispeech960Base
# 10.240.1.2
