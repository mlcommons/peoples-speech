#!/bin/bash

set -euo pipefail

source ./path.sh

lm_url=www.openslr.org/resources/11
work_dir=work_1a

stage=3

mkdir -p $work_dir

#bazel build -c opt --cxxopt='-std=c++14' lingvo:trainer third_party/kaldi/... @openfst//... galvasr2/...

if [ $stage -le 1 ]; then
   local/download_lm.sh $lm_url $work_dir/data/local/lm
fi

langdir=$work_dir/data/lang_char_1gram
if [ $stage -le 2 ]; then
    mkdir -p $work_dir/data/local/dict_char
    # Search for the destination string in librispeech_ctc.py
    gsutil cp local/tokens.txt gs://the-peoples-speech-west-europe/Librispeech/tokens.txt
    galvasr_tokenize_words --in_words_txt $work_dir/data/local/lm/librispeech-vocab.txt \
                           --in_units_txt local/tokens.txt \
                           --out_spelling_txt $work_dir/data/local/dict_char/lexicon.txt \
                           --out_spelling_numbers_txt $work_dir/data/local/dict_char/lexicon_numbers.txt \
                           --out_units_txt $work_dir/data/local/dict_char/units.txt \
                           --space_char "<space>"
    steps/ctc_compile_dict_token.sh --dict-type char \
                                    --space-char "<space>" \
                                    $work_dir/data/local/dict_char \
                                    $work_dir/data/local/lang_char_tmp $work_dir/data/lang_char
    # fstdraw --isymbols=$work_dir/data/lang_char/tokens.txt --osymbols=$work_dir/data/lang_char/tokens.txt $work_dir/data/lang_char/T.fst | dot -Tpdf > T.pdf
    # fstdraw --isymbols=$work_dir/data/lang_char/tokens.txt --osymbols=$work_dir/data/lang_char/words.txt $work_dir/data/lang_char/L.fst | dot -Tpdf > L.pdf
    local/format_lms.sh --src-dir $work_dir/data/lang_char $work_dir/data/local/lm

    fsttablecompose ${langdir}/L.fst $langdir/G.fst | fstdeterminizestar --use-log=true | \
        fstminimizeencoded | fstarcsort --sort_type=ilabel > $langdir/LG.fst || exit 1;
    # Why isn't this determinized and then minimized? Unclear.
    fsttablecompose ${langdir}/T.fst $langdir/LG.fst > $langdir/TLG.fst || exit 1;
    fstconvert --fst_type=const $langdir/TLG.fst $langdir/TLG_const.fst
fi

if [ $stage -le 3 ]; then
    export CUDA_VISIBLE_DEVICES=""
    export OPENBLAS_NUM_THREADS="1"
    export MKL_NUM_THREADS="1"
    # Run training
    # trainer --logdir=gs://the-peoples-speech-west-europe/training_logs/galvez/tpu_ctc_4m \
    #         --mode=sync \
    #         --model=asr.librispeech_ctc.Librispeech960Base \
    #         --logtostderr \
    #         --tpu=grpc://10.240.1.2:8470 \
    #         --job=trainer_client 2>&1 | tee tpu_ctc_4m.log

    # It looks like evaler_dev does run continuously, retrying until a
    # new checkpoint is found.

    trainer --logdir=gs://the-peoples-speech-west-europe/training_logs/galvez/tpu_ctc_4m \
            --mode=sync \
            --model=asr.librispeech_ctc.Librispeech960Base \
            --logtostderr \
            --run_locally=cpu \
            --job=decoder_Dev 2>&1 | tee tpu_ctc_4m_decode_Dev.log

    # Some notes on evaluation:
    # --job=evaler_once_Dev@global_step_count
    # https://github.com/tensorflow/lingvo/blob/b760c994fc9d1b418e0b3f08e00e224449dc0bab/lingvo/tasks/mt/README.md#cluster-configuration
    # cluster_spec="trainer_client=localhost:6007@controller=localhost:6008@decoder_Dev=loalhost:6009"
fi
