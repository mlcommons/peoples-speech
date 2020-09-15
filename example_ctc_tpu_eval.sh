export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

# bazel-bin/lingvo/trainer --logdir=$1 \
#                          --mode=sync \
#                          --model=asr.librispeech_ctc.Librispeech960Base1e4 \
#                          --logtostderr \
#                          --tpu=grpc://10.191.194.74:8470 \
#                          --job=executor_tpu 

# bazel-bin/lingvo/trainer --logdir=gs://the-peoples-speech-west-europe/ag/ctc_librispeech/training_logs/lr1e4/tpu_ctc_1h_lr1e4/ \
bazel-bin/lingvo/trainer --logdir=gs://the-peoples-speech-west-europe/ag/tmp \
                         --mode=sync \
                         --model=asr.librispeech_ctc.Librispeech960Base1e4 \
                         --logtostderr \
                         --run_locally=cpu \
                         # --job=evaler_test
                         

# asr.librispeech_ctc.Librispeech960Base -> lingvo.tasks.asr.params.librispeech_ctc.Librispeech960Base
# 10.240.1.2
