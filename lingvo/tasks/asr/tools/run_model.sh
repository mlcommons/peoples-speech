set -eu

. librispeech_lib.sh

bazel_ run -c opt //lingvo:trainer -- \
       --model=asr.librispeech_ctc.Librispeech960Base \
       --run_locally=cpu \
       --logtostderr \
       --logdir=my_logs/


       # --gcp_project=the-peoples-speech \
       # --tpu_zone= \
       # --tpu=
