
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

bazel-bin/lingvo/trainer -- --logdir=$1 \
                         --mode=sync \
                         --model=asr.librispeech_ctc.Librispeech960BaseLstm7 \
                         --logtostderr \
                         --tpu=grpc://10.55.31.138:8470