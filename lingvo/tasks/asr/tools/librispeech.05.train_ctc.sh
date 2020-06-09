#!/bin/bash

. ./librispeech_lib.sh

set -euo pipefail

cat <<EOF > model_params.txt
    
EOF

bazel_ run //lingvo:trainer \
       --interactive \
       --mode=shell \
       --run_locally=cpu \
       --model=
       --logdir=logs/ctc/
       --model_params_file_override=model_params.txt
