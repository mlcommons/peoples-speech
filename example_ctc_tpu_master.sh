export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

#$1 = tpu/gpu (def tpu)
#$2 = train + eval_dev or train only (def train)
#$3 = logdir (specify if TPU or create a logdir if GPU) (def tpu dir)
#$4 = test_name (def Librispeech960Base)
#$5 = tpu ip

# DATE=$(date '+log_%Y_%m_%d_%H_%M_%S')
DATE=$(date '+log_%Y_%m_%d_%H_%M')

if [ $1 == "gpu" ]
then
    export CUDA_VISIBLE_DEVICES="0"
    # AG TODO: add min and hr to the folder name
    # LOGDIR="/home/anjali/data/librispeech_models/wer/${DATE}"
    LOGDIR="/home/anjali/data/mlcommons/librispeech/models/wer/${DATE}"
else
    export CUDA_VISIBLE_DEVICES=""
    LOGDIR="gs://the-peoples-speech-west-europe/ag/ctc_librispeech/training_logs/$3"
    TPUIP=$5
fi

if [ $2 == "train_evald" ]
then
    # OPERATION="executor_tpu,evaler_dev"
    # OPERATION="tpu_evaluator"
    # OPERATION=trainer_client,evaler_dev
    OPERATION="executor_tpu"
else
    OPERATION="trainer_client"
fi

case $4 in 
    "baseline")
    TEST=Librispeech960Base
    ;;
    "specaugment")
    TEST=Librispeech960BaseSpecAugment
    ;;
    "bidilstm")
    TEST=Librispeech960BaseBidiLstm
    ;;
    "cnn")
    TEST=Librispeech960BaseCnn
    ;;
esac

# ./example_ctc_tpu_lstmcell_2048.sh gs://the-peoples-speech-west-europe/ag/ctc_librispeech/training_logs/lstmcell_2048/
#tpu_ctc_lstmcell_2048_1a 2>&1 | tee tpu_ctc_lstmcell_2048_1a.log

# echo bazel run //lingvo:trainer -- --logdir=${LOGDIR} \
#                          --mode=sync \
#                          --model=asr.librispeech_ctc.${TEST} \
#                          --logtostderr \
#                          --tpu=grpc://${TPUIP}:8470 \
#                          --job-type=$OPERATION 2>&1 | tee ${TEST}_${DATE}.log

bazel run //lingvo:trainer -- --logdir=${LOGDIR} \
                         --mode=sync \
                         --model=asr.librispeech_ctc.${TEST} \
                         --logtostderr \
                         --tpu=grpc://${TPUIP}:8470 \
                         --job=$OPERATION 2>&1 | tee logs/${TEST}_${DATE}.log
                        #  --job-type=$OPERATION 2>&1 | tee ${TEST}_${DATE}.log


