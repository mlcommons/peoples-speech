import os
import json
import argparse

import numpy as np
import editdistance

import nemo.collections.asr as nemo_asr

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])

def main():
    parser = argparse.ArgumentParser(
        description="Subset a tarred local dataset"
    )
    parser.add_argument(
        "--manifest_path",
        required=False,
        type=str,
        help="Manifest file for the files to be transcribed"
    )
    parser.add_argument(
        "--transc_out_path",
        required=True,
        type=str,
        help="Path for the output transcriptions jsonl file"   
    )
    parser.add_argument(
        "--metrics_out_path",
        required=True,
        type=str,
        help="Path for the output metrics file"   
    )
    parser.add_argument(
        "--asr_ckpt_path",
        type=str,
        help="Local path to a .ckpt NeMo ASR model checkpoint",
        default=None
    )
    parser.add_argument(
        "--asr_batch_size",
        required=False,
        type=int,
        help="Inference batch size for acustic model",
        default=16
    )
    parser.add_argument(
        "--kenlm_model_path",
        required=True,
        type=str,
        help="A KENLM model file (https://github.com/kpu/kenlm)"
    )
    parser.add_argument(
        "--beam_width",
        required=False,
        type=int,
        default=1024,
        help="Beam width for beam search"
    )
    parser.add_argument(
        "--beam_search_alpha",
        required=False,
        type=float,
        default=1.0,
        help="Beam search alpha parameter"
    )
    parser.add_argument(
        "--beam_search_beta",
        required=False,
        type=float,
        default=1.0,
        help="Beam search alpha parameter"
    )
    args = parser.parse_args()
    try:
        from ctc_decoders import Scorer, ctc_beam_search_decoder_batch
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "BeamSearchDecoderWithLM requires the installation of ctc_decoders "
            "from NeMo/scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh"
        )
    asr_model = nemo_asr.models.ctc_models.EncDecCTCModel.load_from_checkpoint(
        checkpoint_path=args.asr_ckpt_path
    )
    samples = []
    with open(args.manifest_path, "r") as manifest:
        for str_sample in manifest:
            sample = json.loads(str_sample)
            samples.append(sample)
    asr_logits = asr_model.transcribe(
        [s["audio_filepath"] for s in samples],
        batch_size=args.asr_batch_size,
        logprobs=True
    )
    # asr_probs[i][j] := 
    asr_probs = [softmax(logits) for logits in asr_logits]
    vocab = asr_model.decoder.vocabulary
    scorer = Scorer(
        args.beam_search_alpha,
        args.beam_search_beta,
        model_path=args.kenlm_model_path,
        vocabulary=vocab
    )
    transcriptions = ctc_beam_search_decoder_batch(
        probs_split=asr_probs,
        vocabulary=vocab,
        beam_size=args.beam_width,
        ext_scoring_func=scorer,
        num_processes=max(os.cpu_count(), 1)
    )
    transcriptions = [t[0][1] for t in transcriptions]
    char_scores = 0
    chars = 0
    word_scores = 0
    words = 0
    with open(args.transc_out_path, "w") as transc_file:
        for sample, transc in zip(samples, transcriptions):
            # Write transcription
            sample["transcription"] = transc
            transc_file.write(str_sample + "\n")
            # Track performance metrics
            char_scores += editdistance.eval(list(transc), list(sample["text"]))
            chars += len(sample["text"])
            word_scores += editdistance.eval(transc.split(), sample["text"].split())
            words += len(sample["text"].split())
            str_sample = json.dumps(sample)
    
    with open(args.metrics_out_path, "w") as metrics_file:
        metrics = {
            "wer": 1.0 * word_scores / words,
            "cer": 1.0 * char_scores / chars
        }
        str_metrics = json.dumps(metrics, indent=1)
        metrics_file.write(str_metrics)

if __name__ == "__main__":
    main()