import os
import uuid
import glob
import json
import errno
import shutil
import tarfile
import argparse
import warnings
import itertools
import contextlib
from collections import ChainMap

import torch
import numpy as np
import editdistance
from tqdm import tqdm
import pytorch_lightning as pl
import torch.multiprocessing as mp

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel

TMP_DIR = os.path.join(os.getcwd(), "transcribe_tmp_dir")

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])

def _asr_on_filepaths(
    audio_filepaths, audio_dir, pretrained_model, asr_batch_size, asr_ckpt_path,
    device, asr_model, logprobs, **kw
    ):
    # Load acustic model
    if asr_model is None:
        if asr_ckpt_path:
            warnings.warn("Models loaded from a .ckpt run on CPU")
            # TODO: Infer this kind of model in GPU
            asr_model = EncDecCTCModel.load_from_checkpoint(
                checkpoint_path=asr_ckpt_path,
                map_location=device
            )
        else:
            asr_model = ASRModel.from_pretrained(
                model_name=pretrained_model,
                map_location=device
            )
        trainer = pl.Trainer(gpus=int(device != "cpu"))
        asr_model.set_trainer(trainer)
        asr_model = asr_model.eval()

    # Transcribe
    if audio_dir:
        full_filepaths = [os.path.join(audio_dir, p) for p in audio_filepaths]
    else:
        full_filepaths = audio_filepaths
    @contextlib.contextmanager
    def autocast():
        yield
    with autocast():
        with torch.no_grad():
            preds = asr_model.transcribe(
                full_filepaths,
                batch_size=asr_batch_size,
                logprobs=logprobs
            )
    filepath_to_pred = {}
    for filepath, pred in zip(audio_filepaths, preds):
        filepath_to_pred[filepath] = pred
    return filepath_to_pred, asr_model

def _asr_on_tarpaths_chunk(chunk_args):
    tarpaths_chunk = chunk_args["tarpaths_chunk"]
    transc_args = chunk_args["transc_args"]
    device = transc_args["device"]
    tmp_dir = TMP_DIR + "_" + str(uuid.uuid1())
    asr_model = None
    filepath_to_pred = {}
    for audio_tarpath in tqdm(tarpaths_chunk, desc="Tar files in chunk: "):
        # Extract audio files into tmp_dir
        os.makedirs(tmp_dir)
        try:
            with tarfile.open(audio_tarpath) as audio_tarfile:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner) 
                    
                
                safe_extract(audio_tarfile, path=tmp_dir)
        except OSError as exc:
            # Skip long-named audio files
            if exc.errno == errno.ENAMETOOLONG:
                with tarfile.open(audio_tarpath) as audio_tarfile:
                    members = audio_tarfile.getmembers()
                    healthy_members = [
                        m for m in members
                        if len(m.name) <= os.pathconf("/", "PC_NAME_MAX")
                    ]
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner) 
                        
                    
                    safe_extract(audio_tarfile, path=tmp_dir, members=healthy_members)
                    warnings.warn(
                        f"Some audio files were dropped from {audio_tarpath} "
                        "because they have long file names"
                    )
            else:
                raise  # re-raise previously caught exception
        audio_filepaths = os.listdir(tmp_dir)

        # Transcribe
        transc_args["device"] = device
        transc_args["audio_filepaths"] = audio_filepaths
        transc_args["audio_dir"] = tmp_dir
        transc_args["asr_model"] = asr_model
        local_filepath_to_pred, asr_model = _asr_on_filepaths(**transc_args)
        filepath_to_pred.update(local_filepath_to_pred)
        shutil.rmtree(tmp_dir)
        os.rmdir(tmp_dir)
    return filepath_to_pred

def _decode_with_lm(
    filepath_to_logprobs, kenlm_model_path, vocab, beam_width, beam_search_alpha,
    beam_search_beta
    ):
    filepaths = list(filepath_to_logprobs.keys())
    asr_logprobs = [filepath_to_logprobs[p] for p in filepaths]
    from ctc_decoders import Scorer, ctc_beam_search_decoder_batch
    scorer = Scorer(
        beam_search_alpha,
        beam_search_beta,
        model_path=kenlm_model_path,
        vocabulary=vocab
    )
    asr_probs = [softmax(logits) for logits in asr_logprobs]
    transcriptions = ctc_beam_search_decoder_batch(
        probs_split=asr_probs,
        vocabulary=vocab,
        beam_size=beam_width,
        ext_scoring_func=scorer,
        num_processes=max(os.cpu_count(), 1)
    )
    transcriptions = [t[0][1] for t in transcriptions]
    filepath_to_transc = {}
    for filepath, transc in zip(filepaths, transcriptions):
        filepath_to_transc[filepath] = transc
    return filepath_to_transc

def _score_and_write_transcriptions(samples, filepath_to_transc, out_dir, args):
    transc_out_path = os.path.join(out_dir, "transc-manifest.jsonl")
    metrics_out_path = os.path.join(out_dir, "transc-metrics.json")
    args_out_path = os.path.join(out_dir, "cmd-args.json")
    total_char_dist = 0
    total_chars = 0
    total_word_dist = 0
    total_words = 0
    with open(transc_out_path, "w") as transc_file:
        print("Writting scored transcriptions")
        for sample in tqdm(samples):
            transc = filepath_to_transc.get(sample["audio_filepath"])
            if transc is None:
                continue
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
        str_args = json.dumps(args, indent=1)
        args_file.write(str_args)


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
        "--audio_dir",
        required=False,
        type=str,
        default=None,
        help="If the audios are untarred, provide directory that contains audio paths"
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
        "--beam_search_alphas",
        required=False,
        type=str,
        default="1.0",
        help="Beam search alpha parameter"
    )
    parser.add_argument(
        "--beam_search_betas",
        required=False,
        type=str,
        default="1.0",
        help="Beam search alpha parameter"
    )
    args = parser.parse_args()

    # Verify LM decoders installation if necessary
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

    os.makedirs(args.out_dir)

    # Read samples into CPU RAM
    samples = []
    with open(args.manifest_path, "r") as manifest:
        for str_sample in manifest:
            sample = json.loads(str_sample)
            samples.append(sample)
    
    # Get ASR transcription or logprobs
    transc_args = vars(args)
    transc_args["asr_model"] = None
    transc_args["logprobs"] = args.kenlm_model_path is not None
    if args.tarfiles_glob:
        # Split the tarpaths accross the available GPUs
        audio_tarpaths = glob.glob(args.tarfiles_glob)
        n_gpus = torch.cuda.device_count()
        print(f"Transcribing on {n_gpus} GPUs")
        if n_gpus < 1:
            # TODO: Enable tarfile transcription on CPU
            raise NotImplementedError("Tarfile transcription on CPU not implemented")
        n_tarpaths = len(audio_tarpaths)
        chunk_size = int(np.ceil(n_tarpaths / n_gpus))
        chunks = []
        for i in range(n_gpus):
            tarpaths_chunk = audio_tarpaths[(i * chunk_size):((i+1) * chunk_size)]
            transc_args["device"] = f"cuda:{i}"
            chunks.append({
                "tarpaths_chunk": tarpaths_chunk,
                "transc_args": transc_args
            })
        spawn_ctx = mp.get_context("spawn")
        with spawn_ctx.Pool(n_gpus) as pool:
            filepath_to_pred_list = pool.map(_asr_on_tarpaths_chunk, chunks)
        filepath_to_pred = ChainMap(*filepath_to_pred_list)
        del filepath_to_pred_list
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        transc_args = vars(args)
        transc_args["device"] = device
        transc_args["audio_filepaths"] = [s["audio_filepath"] for s in samples]
        transc_args["audio_dir"] = args.audio_dir
        transc_args["asr_model"] = None
        filepath_to_pred, asr_model = _asr_on_filepaths(**transc_args)

    if not transc_args["logprobs"]:
        _score_and_write_transcriptions(
            samples=samples,
            filepath_to_transc=filepath_to_pred,
            out_dir=args.out_dir,
            args=vars(args)
        )
    else:
        if args.beam_search_alphas is None or args.beam_search_betas is None:
            raise ValueError(
                "Requested LM transcription but either `beam_search_alphas` or "
                "`beam_search_betas` is not set"
            )
        beam_search_alphas = [float(a) for a in args.beam_search_alphas.split(",")]
        beam_search_betas = [float(b) for b in args.beam_search_betas.split(",")]
        grid = itertools.product(beam_search_alphas, beam_search_betas)
        for beam_search_alpha, beam_search_beta in grid:
            filepath_to_transc = _decode_with_lm(
                filepath_to_logprobs=filepath_to_pred,
                kenlm_model_path=args.kenlm_model_path,
                vocab=asr_model.decoder.vocabulary,
                beam_width=args.beam_width,
                beam_search_alpha=beam_search_alpha,
                beam_search_beta=beam_search_beta
            )
            grid_item_id = f"a={beam_search_alpha}_b={beam_search_beta}"
            out_dir = os.path.join(args.out_dir, grid_item_id)
            local_args = vars(args)
            local_args.update({
                "beam_search_alpha": beam_search_alpha,
                "beam_search_beta": beam_search_beta
            })
            os.makedirs(out_dir)
            _score_and_write_transcriptions(
                samples=samples,
                filepath_to_transc=filepath_to_transc,
                out_dir=out_dir,
                args=local_args               
            )

if __name__ == "__main__":
    main()