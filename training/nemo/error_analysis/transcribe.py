import os
import glob
import json
import tarfile
import argparse
import warnings
import contextlib

import torch
import numpy as np
import editdistance
from tqdm import tqdm
import pytorch_lightning as pl

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel

TMP_DIR = os.path.join(os.getcwd(), "transcibe_tmp_dir")

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])

def main():
    parser = argparse.ArgumentParser(
        description="Subset a tarred local dataset"
    )
    parser.add_argument(
        "--manifest_path",
        required=True,
        type=str,
        help="Manifest file for the files to be transcribed"
    )
    parser.add_argument(
        "--tarfiles_glob",
        required=False,
        type=str,
        default=None,
        help="If the audios are tarred, provide a glob pattern of the .tar paths"
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Path for the output transcriptions and metrics"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="stt_en_citrinet_1024_gamma_0_25",
        required=False,
        help="String key which will be used by NGC to find the module"
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
        required=False,
        type=str,
        help="A KENLM model file (https://github.com/kpu/kenlm)",
        default=None
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
    transc_out_path = os.path.join(args.out_dir, "transc-manifest.jsonl")
    metrics_out_path = os.path.join(args.out_dir, "transc-metrics.json")
    args_out_path = os.path.join(args.out_dir, "cmd-args.json")

    # Verify LM transcription imports if needed
    if args.kenlm_model_path:
        try:
            from ctc_decoders import Scorer, ctc_beam_search_decoder_batch
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "BeamSearchDecoderWithLM requires the installation of "
                "ctc_decoders from NeMo/scripts/asr_language_modeling/ngram_lm/"
                "install_beamsearch_decoders.sh"
            )
    else:
        warnings.warn("No LM given, will do greedy decoding with ASR model")

    # Setup GPU
    gpu_available = torch.cuda.is_available()
    device = torch.device(f'cuda:{0}' if gpu_available else 'cpu')
    
    # Load acustic model
    if args.asr_ckpt_path:
        warnings.warn("Models loaded from a .ckpt run on CPU")
        # TODO: Infer this kind of model in GPU
        asr_model = EncDecCTCModel.load_from_checkpoint(
            checkpoint_path=args.asr_ckpt_path,
            map_location=device
        )
    else:
        asr_model = ASRModel.from_pretrained(
            model_name=args.pretrained_model,
            map_location=device
        )
    trainer = pl.Trainer(gpus=int(gpu_available))
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    # Read samples into CPU RAM
    samples = []
    with open(args.manifest_path, "r") as manifest:
        for str_sample in manifest:
            sample = json.loads(str_sample)
            samples.append(sample)
    
    # Transcribe
    @contextlib.contextmanager
    def autocast():
        yield
    if args.kenlm_model_path:
        if args.tarfiles_glob:
            raise NotImplementedError(
                "Transcription from .tar with LM not implemented"
            )
        asr_logits = asr_model.transcribe(
            [s["audio_filepath"] for s in samples],
            batch_size=args.asr_batch_size,
            logprobs=True
        )
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
    else:
        if args.tarfiles_glob:
            filepath_to_transc = {}
            audio_tarpaths = glob.glob(args.tarfiles_glob)
            os.makedirs(TMP_DIR, exist_ok=True)
            for audio_tarpath in audio_tarpaths:
                tmp_dir_files = glob.glob(os.path.join(TMP_DIR, "*"))
                for tmp_file in tmp_dir_files:
                    os.remove(tmp_file)
                with tarfile.open(audio_tarpath) as audio_tarfile:
                    audio_tarfile.extractall(path=TMP_DIR)
                    full_filepaths = glob.glob(os.path.join(TMP_DIR, "*"))
                # Transcribe audios in tarfile
                with autocast():
                    with torch.no_grad():
                        transcs = asr_model.transcribe(
                            full_filepaths,
                            batch_size=args.asr_batch_size,
                            logprobs=False
                        )
                for full_filepath, transc in zip(full_filepaths, transcs):
                    audio_filepath = os.path.basename(full_filepath)
                    filepath_to_transc[audio_filepath] = transc
            transcriptions = [
                filepath_to_transc[s["audio_filepath"]] for s in samples
            ]
        else:
            with autocast():
                with torch.no_grad():
                    transcriptions = asr_model.transcribe(
                        [s["audio_filepath"] for s in samples],
                        batch_size=args.asr_batch_size,
                        logprobs=False
                    )

    # Score and write transcriptions
    total_char_dist = 0
    total_chars = 0
    total_word_dist = 0
    total_words = 0
    os.makedirs(args.out_dir)
    with open(transc_out_path, "w") as transc_file:
        print("Writting scored transcriptions")
        for sample, transc in tqdm(zip(samples, transcriptions)):
            # Track performance metrics
            char_dist = editdistance.eval(list(transc), list(sample["text"]))
            total_char_dist += char_dist
            n_chars = len(sample["text"])
            total_chars += n_chars
            word_dist = editdistance.eval(transc.split(), sample["text"].split())
            total_word_dist += word_dist
            n_words = len(sample["text"].split())
            total_words += n_words
            # Write transcription
            sample["pred_text"] = transc
            sample["cer"] = char_dist / n_chars
            sample["wer"] = word_dist / n_words
            str_sample = json.dumps(sample)
            transc_file.write(str_sample + "\n")
    with open(metrics_out_path, "w") as metrics_file:
        metrics = {
            "cer": 1.0 * total_char_dist / total_chars,
            "wer": 1.0 * total_word_dist / total_words
        }
        str_metrics = json.dumps(metrics, indent=1)
        metrics_file.write(str_metrics)
    
    # Write command line arguments
    with open(args_out_path, "w") as args_file:
        dict_args = vars(args)
        str_args = json.dumps(dict_args, indent=1)
        args_file.write(str_args)

if __name__ == "__main__":
    main()