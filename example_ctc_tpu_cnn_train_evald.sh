
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

bazel run //lingvo:trainer -- --logdir=$1 \
                         --mode=sync \
                         --model=asr.librispeech_ctc.Librispeech960BaseCnn \
                         --logtostderr \
                         --tpu=grpc://10.228.39.162:8470 \
                         --job-type=executor_tpu,evaler_dev