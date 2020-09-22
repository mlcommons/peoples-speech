#making GPU_id 0 visible to TF, for TPU - unset/change to ""
export CUDA_VISIBLE_DEVICES="0"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

LOGDIR="/home/anjali/data/librispeech_models/wer/train_1e4/"   
bazel run //lingvo:trainer -- --logdir=${LOGDIR} \
                                --mode=sync \
                                --model=asr.librispeech_ctc.Librispeech960Base \
                                --logtostderr \
                                --job=evaler_dev \
                                --run_locally=gpu 2>&1 | tee -a log #-a appends to log file clled log
                         

# asr.librispeech_ctc.Librispeech960Base -> lingvo.tasks.asr.params.librispeech_ctc.Librispeech960Base
# 10.240.1.2
