#making GPU_id 0 visible to TF, for TPU - unset/change to ""
export CUDA_VISIBLE_DEVICES="0"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

# run the cmd below in the shell and get the curr date(o/p)
#DATE=$(date '+%m%d')
DATE=$(date '+log_%Y_%m_%d_%H_%M_%S')
# AG TODO: add min and hr to the folder name
LOGDIR="/home/anjali/data/librispeech_models/wer/${DATE}"

echo "Resetting and reusing ${LOGDIR}" | tee log
rm -rf $LOGDIR

bazel run //lingvo:trainer -- --logdir=${LOGDIR} \
                         --mode=sync \
                         --model=asr.librispeech_ctc.Librispeech960Base \
                         --logtostderr \
                         --run_locally=gpu 2>&1 | tee -a log #-a appends to log file clled log

                         # --tpu=grpc://10.240.1.2:8470 \
                         # --job=executor_tpu


# asr.librispeech_ctc.Librispeech960Base -> lingvo.tasks.asr.params.librispeech_ctc.Librispeech960Base
# 10.240.1.2
