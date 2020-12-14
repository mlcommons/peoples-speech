#! /bin/bash
# set -x

model=$1

GS_BASE="gs://the-peoples-speech-west-europe/PeoplesSpeech/ag_training"
HOME_BASE="${HOME}/data/PeoplesSpeech/models"
cd $HOME_BASE

gsutil ls -r "${GS_BASE}/${model}" > all_files

function download_list () {
  pattern=$1
  grep $pattern all_files | while read gspath 
  do
    gsutil cp ${gspath} ".${gspath/${GS_BASE}}"
  done
}

function download_ckpt () {
  pattern="checkpoint$"
  grep $pattern all_files | while read gspath 
  do
    ckpt_path=".${gspath/${GS_BASE}}"
    gsutil cp ${gspath} ${ckpt_path}
    ckpt=$(grep "^model_checkpoint_path" ${ckpt_path} | cut -f2 -d\" )
    model_path=$(dirname ${ckpt_path})
    gs_model_path=$(dirname ${gspath})
    gsutil cp -r ${gs_model_path}/${ckpt}* ${model_path}
  done
}


download_list "eval_tpu_dev/events*"
download_list "eval_tpu_train/events*"
download_list "train_train/events*"
download_list "train_params.txt"
download_ckpt 
