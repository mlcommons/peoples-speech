export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"


HOME_BASE="${HOME}/data/PeoplesSpeech/models"
GS_BASE="gs://the-peoples-speech-west-europe/PeoplesSpeech/ag_training"

DATE=$(date '+log_%Y_%m_%d_%H')
CLS=$3
FLDRDATE=$4

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

    DEVICE="--tpu=grpc://${TPUIP}:8470"

elif [ $1 == "gpu" ]; then
    export CUDA_VISIBLE_DEVICES="0"
    DEVICE="--run_locally=gpu"
    LOGDIR="${HOME_BASE}/${FLDRDATE}"
else
    export CUDA_VISIBLE_DEVICES=""
    LOGDIR="${HOME_BASE}/${FLDRDATE}"
    DEVICE="--run_locally=cpu"
fi

if [ $2 == "decode" ]; then
    OPERATION="decoder_Dev"
    LOGDIR="${LOGDIR}/${CLS}"

elif [ $2 == "trainer" ]; then
    OPERATION="trainer_client"
else
    OPERATION="executor_tpu"
fi

bazel run //lingvo:trainer -- --logdir=${LOGDIR} \
    --mode=sync \
    --model=asr.peoplesspeech_ctc.${CLS} \
    --logtostderr \
    ${DEVICE} \
    --job=$OPERATION 2>&1 | tee logs/${CLS}_${DATE}.log
