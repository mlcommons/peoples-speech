
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

bazel run //lingvo:trainer -- --logdir=$1 \
                         --mode=sync \
                         --model=asr.librispeech_ctc.Librispeech960Base \
                         --logtostderr \
                         --tpu=grpc://10.174.188.74:8470 \
                         --job-type=executor_tpu,evaler_dev