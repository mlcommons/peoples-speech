export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

#$1 = tpu/gpu (def tpu)
#$2 = train + eval_dev or train only (def train)
#$3 = logdir (specify if TPU or create a logdir if GPU) (def tpu dir)
#$4 = test_name (def Librispeech960Base)
#$5 = tpu ip

# DATE=$(date '+log_%Y_%m_%d_%H_%M_%S')
DATE=$(date '+log_%Y_%m_%d_%H')

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

if [ $2 == "decode" ]
then
    # OPERATION="executor_tpu,evaler_dev"
    # OPERATION="tpu_evaluator"
    # OPERATION=trainer_client,evaler_dev
    # OPERATION="executor_tpu"
    OPERATION="decoder_dev"
else
    OPERATION="executor_tpu"
fi

case $4 in 
    "baseline")
    TEST=Librispeech960Grapheme
    ;;
    "wpm_spec")
    TEST=Librispeech_Wpm_SpecAug_InptStack
    ;;
    "graph_spec")
    TEST=Librispeech_Grphm_SpecAug_InptStack
    ;;
    "cnn")
    TEST=Librispeech960BaseCnn
    ;;
    "bidi")
    TEST=Librispeech960BaseBidiLstm
    ;;
    "wpm")
    TEST=Librispeech960Wpm
    ;;
    "cnn_wpm_spec")
    TEST=Librispeech960Base_Cnn_Wpm_Spec
    ;;
    "cnn_wpm")
    TEST=Librispeech_Cnn_Wpm
    ;;
    "spec_1536")
    TEST=Librispeech_Grphm_1536_SpecAug_InptStack
    ;;
    "do_spec_6x1024")
    TEST=Librispeech_Grphm_DO_SpecAug_InptStack_6x1024
    ;;
    "do_spec_5x1536")
    TEST=Librispeech_Grphm_DO_SpecAug_InptStack_5x1536
    ;;
    "do_spec")
    TEST=Librispeech_Grphm_DO_SpecAug_InptStack
    ;;
esac

# ./example_ctc_tpu_lstmcell_2048.sh gs://the-peoples-speech-west-europe/ag/ctc_librispeech/training_logs/lstmcell_2048/
#tpu_ctc_lstmcell_2048_1a 2>&1 | tee tpu_ctc_lstmcell_2048_1a.log
# ./example_ctc_tpu_master.sh tpu train 1006/lr1e5_wpm_spec_ascii_space wpm_spec 10.240.1.10 
# bazel-bin/lingvo/trainer --logdir=gs://the-peoples-speech-west-europe/ag/ctc_librispeech/training_logs/1006/lr1e5_wpm_spec_ascii_space --mode=sync --model=asr.librispeech_ctc.Librispeech_Wpm_SpecAug_InptStack --logtostderr --run_locally=cpu --job=decoder_dev


# ./example_ctc_tpu_master.sh tpu trainer_client 1004/wpm_spec_ascii_space wpm_spec 10.240.1.2
# bazel-bin/lingvo/trainer --logdir=gs://the-peoples-speech-west-europe/ag/ctc_librispeech/training_logs/1004/wpm_spec_ascii_space --mode=sync --model=asr.librispeech_ctc.Librispeech_Wpm_SpecAug_InptStack --logtostderr --run_locally=cpu --job=decoder_dev

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


# ../ctpu st --name=ag-test-tpu --details
# ../ctpu up -name ag-test-tpu -tpu-only -tpu-size v3-8 -tf-version 2.2 --preemptible
# ./example_ctc_tpu_master.sh tpu train 1006/baseline_new_encoder_inp_stacking baseline 10.193.173.186
