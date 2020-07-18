#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ROOT=/tmp/librispeech
ROOT=gs://the-peoples-speech-west-europe/Librispeech

# From:
# http://www.openslr.org/12/
SOURCE=http://www.openslr.org/resources/12

# If in China, use this mirror:
# http://cn-mirror.openslr.org/resources/12

# conda activate galvasr

# My home directory in on an NFS mount, which bazel dislikes, so I need to do this.
bazel_() {
  bazel --output_user_root=/home/ws15dgalvez/dgalvez-b02/.cache/bazel "$@"
}

qsub_run() {
  cat <<EOF  > qsub_script.sh 
#!/bin/bash
#$ -cwd
#$ -j yes
#$ -l mem_free=5G,ram_free=5G,arch=*64*
#$ -v PATH

. ./librispeech_lib.sh

export CUDA_VISIBLE_DEVICES=""
source "$($CONDA_EXE info --base)/etc/profile.d/conda.sh"
conda activate galvasr

$@
EOF

qsub qsub_script.sh
}

# Never instantiate a cuda context.
export CUDA_VISIBLE_DEVICES=""
