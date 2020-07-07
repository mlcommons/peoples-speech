# python -m lingvo.train --ctc_model=PLPCoefficients \
#        --abc 1 \
#        --xyz 2

# bazel run //lingvo/train --model

export CUDA_VISIBLE_DEVICES=""

bazel-bin/lingvo/trainer --logdir=librispeech_logs2 \
                         --mode=sync --run_locally=cpu \
                         --model=asr.librispeech_ctc.Librispeech960Base \
                         --logtostderr
