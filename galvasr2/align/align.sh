#!/bin/bash

debug_loglevel=10

for aggressiveness in $(seq 0 3); do
    out_dir=sample_alignment_data/output-create-buffer/${aggressiveness}
    mkdir -p ${out_dir}
    # /usr/lib/linux-tools-4.9.0-12/perf record -o /perf.out 
    python galvasr2/align/align.py \
       --stt-model-dir third_party/DSAlign/models/en \
       --force \
       --audio sample_alignment_data/Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49.mp3 \
       --script sample_alignment_data/Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49.txt \
       --aligned ${out_dir}/aligned.json \
       --tlog ${out_dir}/transcript.log \
       --audio-vad-aggressiveness $aggressiveness \
       --loglevel ${debug_loglevel} \
       --output-wng --output-jaro_winkler --output-editex --output-levenshtein --output-mra --output-hamming --output-cer --output-wer --output-sws --output-mlen --output-tlen

    exit 0

    #    --align-shrink-fraction 0

done


# I need to librarize alignment.

# I the VAD stage

# The ASR stage

# The alignment and postprocessing stages.

# Need to investigate VAD lengths. Possibly change it so that the
# chunks are at least 15 seconds in length.

# Use a lower WER threshold for files ending in "asr.srt"

# Enable per-document language models.

#
# DEBUG:root:Fragment 302 aligned with WER: 66.67 CER: 88.89 hamming: 0.00 mra: 66.67 levenshtein: 50.00 editex: 59.38 jaro_winkler: 62.96 WNG: 48.86 SWS: 50.00
# DEBUG:root:- T:           "you know time to"
# DEBUG:root:- O:  giving hi|m time to| appears o

# May want to recompile from source in order to get a speedup in speech-to-text process.
# Helpful because less TPU nonsense!

# 2020-12-03 07:16:52.045130: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA


# python galvasr2/align/align.py \
#        --stt-model-dir third_party/DSAlign/models/en \
#        --force \
#        --audio sample_alignment_data/Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49.mp3 \
#        --script sample_alignment_data/Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49.txt \
#        --aligned sample_alignment_data/aligned.json \
#        --tlog sample_alignment_data/transcript.log \
#        --audio-vad-aggressiveness 2 \
#        --loglevel ${debug_loglevel} \
# --align-shrink-fraction 0 --align-similarity-algo "levenshtein" --output-wng --output-jaro_winkler --output-editex --output-levenshtein --output-mra --output-hamming --output-cer --output-wer --output-sws --output-mlen --output-tlen --output-pretty



# If I get the right outputs from this, then I should be able to just
# analyze it in SparkSQL. Minimize the changes I make to the alignment code overall.

# I need to rerun the STT to get this to work...hmm. Slow!!!!
# Challenging. I could script it, though. Need to make I output
# everything I want, though.

# level 3 and 2:
# Used Dropped Total
# 603  299     902

       
#       --per-document-lm
       
# --align-shrink-fraction 0 --align-similarity-algo "levenshtein" --output-wng --output-jaro_winkler --output-editex --output-levenshtein --output-mra --output-hamming --output-cer --output-wer --output-sws --output-mlen --output-tlen --output-pretty \
