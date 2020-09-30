export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

bazel run //lingvo:trainer -- --logdir=$1 \
                         --mode=sync \
                         --model=asr.librispeech_ctc.Librispeech960Base \
                         --logtostderr \
                         --tpu=grpc://10.240.1.2:8470 \
                         --job=executor_tpu

# asr.librispeech_ctc.Librispeech960Base -> lingvo.tasks.asr.params.librispeech_ctc.Librispeech960Base
# 10.240.1.2
