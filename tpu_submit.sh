export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

#$1 = tpu/gpu (def tpu)
#$2 = train + eval_dev or train only (def train)
#$3 = class_name (def Librispeech960Base)

HOME_BASE="/home/anjali/data/mlcommons/librispeech/models/wer"
# GS_BASE="gs://the-peoples-speech-west-europe/ag/ctc_librispeech/training_logs"
GS_BASE="gs://the-peoples-speech-west-europe/"

DATE=$(date '+log_%Y_%m_%d_%H')
# FLDRDATE=$(date '+%m%d/%H%M')
# FLDRDATE=$(date '+%m%d')
FLDRDATE=$4
CLS=$3

# if tpu- in $1
if [[ "$1" == *"-tpu"* ]]; then
    LOGDIR="${GS_BASE}/${FLDRDATE}/${CLS}"

    name=$1
    ip_addr=$(ctpu st -name ${name} --details | grep "TPU IP" | grep -oP "10.*")

    if [ -z "$ip_addr" ]; then
        echo "Couldnt find TPU, creating a new one"
        ctpu up -name ${name} -tpu-only -tpu-size v3-8 -tf-version 2.2
        ctpu st -name ${name} --details | grep "TPU IP" | grep -oP "10.*"
        ip_addr=$(ctpu st -name ${name} --details | grep "TPU IP" | grep -oP "10.*")
    fi

    TPUIP=$ip_addr

elif [ $1 == "gpu" ]; then
    export CUDA_VISIBLE_DEVICES="0"
    # LOGDIR="/home/anjali/data/librispeech_models/wer/${DATE}"
    LOGDIR="${HOME_BASE}/${DATE}"
else
    export CUDA_VISIBLE_DEVICES=""
    LOGDIR="${GS_BASE}/${FLDRDATE}/${CLS}"
    TPUIP=$5
fi

if [ $2 == "decode" ]; then
    OPERATION="decoder_dev"
elif [ $2 == "trainer" ]; then
    OPERATION="trainer_client"
else
    OPERATION="executor_tpu"
fi

bazel run //lingvo:trainer -- --logdir=${LOGDIR} \
    --mode=sync \
    --model=asr.librispeech_ctc.${CLS} \
    --logtostderr \
    --tpu=grpc://${TPUIP}:8470 \
    --job=$OPERATION 2>&1 | tee logs/${CLS}_${DATE}.log

    # ./submit.sh tpu_name executor class_model_name fldr_name
    # ./submit.sh ag-tpu1-1019 executor Grphm_DO_SpecAug_InptStack_6x1024
    # ./example_ctc_tpu_master.sh ag-tpu3-1019 executor Grphm_DO_SpecAug_ConvStack_6x1024 1020
    # ./example_ctc_tpu_master.sh ag-tpu2-1019 executor Grphm_DO_SpecAug_InptStack_6x512Bidi 1020