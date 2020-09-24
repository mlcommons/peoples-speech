
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

bazel run //lingvo:trainer -- --logdir=$1 \
                         --mode=sync \
                         --model=asr.librispeech_ctc.Librispeech960BaseLstm2048 \
                         --logtostderr \
                         --tpu=grpc://10.201.226.122:8470 \
1                        --job-type=executor_tpu