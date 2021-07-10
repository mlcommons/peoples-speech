

from urllib.parse import urlparse
from google.cloud import storage
import numpy as np
import resampy
import tensorflow as tf
import tensorflow_io as tfio
import urllib.request
import itertools
import io
import jsonlines

import galvasr2.yamnet.params as yamnet_params
import galvasr2.yamnet.yamnet as yamnet_model

import os

from argparse import ArgumentParser

import config

import logging

logger = logging.getLogger(__name__)

def main():
    config = parse_arguments()
    setup_logging(config)
    run_inference(config)

def parse_arguments():
    parser = ArgumentParser("Run YAMNET on a set of audio files.")
    parser.add_argument("-i", "--input-path", default="gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/training_set", help="Path to yamnet dataset.")
    parser.add_argument("-o", "--output-path", default="results.jsonl", help="The path to save the results.")
    parser.add_argument("-c", "--config-file-path", default=".json", help="The path to the config file.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print out debug messages.")
    parser.add_argument("-vi", "--verbose-info", default=False, action="store_true", help="Print out info messages.")

    args = parser.parse_args()
    arguments = vars(args)

    config = setup_config(arguments)

    return config

def run_inference(config):
    model, classes, params = load_model(config)

    dataset = get_dataset(config)

    run_model_on_dataset(model, classes, params, dataset, config)

def get_dataset(config):
    # Replace with a single file to start with?
    files = tf.data.Dataset.list_files(
        os.path.join(config["input_path"],
                     "Our_Community_Cares_Camp_Public_Service_Announcement/Our_Community_Cares_Camp_Public_Service_Announcement.mp3"))
    # files = tf.data.Dataset.list_files(os.path.join(config["input_path"], "**/*.mp3"))
    def get_audio_and_rate_tensors(file_name):
        io_tensor = tfio.audio.AudioIOTensor(file_name, dtype=tf.float32)
        return (io_tensor.to_tensor(), io_tensor.rate)
    ds = files.map(get_audio_and_rate_tensors)
    # ds = files.map(tf.io.read_file)
    # ds = ds.map(tfio.audio.decode_mp3)
    def split_into_chunks(audio_tensor, sampling_rate):
        audio_tensor  sampling_rate * config['seconds_per_chunk']
        tf.split(, axis=0)
    ds = ds.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def load_model(config):
    logger.debug("Loading model...")

    weights = load_weights(config)

    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights(weights)

    yamnet_classes = yamnet_model.class_names(os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv"))

    return yamnet, yamnet_classes, params

def load_weights(config):
    download_path = "https://storage.googleapis.com/audioset/yamnet.h5"

    target_path = "/tmp/yamnet/yamnet.h5"

    download(download_path, target_path)

    return target_path

def download(url, path):
    logger.debug("Downloading from " + url + " to " + path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    logger.debug("Download success")

def run_model_on_dataset(yamnet, classes, params, dataset, config):
    with jsonlines.open(config["output_path"], mode='w') as writer:
        for batch in dataset:
            # How do we split the audio fwave forms?
            items = split_into_items(batch, config)
            logger.debug("chunks" + str(len(items)))
            for index, item in enumerate(items):
                results = run_model_on_batch(yamnet, classes, params, item)
                print_results(writer, results, classes, index, config)

def split_into_items(pair, config):
    # Ugh
    batch = pair[:-1]
    sr = int(pair[-1])
    chunk_size = int(float(config["seconds_per_chunk"]) * sr)

    array = batch.numpy()

    sample_count = array.shape[-1]

    logger.debug("total samples " + str(array.shape))

    items = []

    chunks = (sample_count + chunk_size - 1) // chunk_size

    for chunk in range(chunks):
        start = chunk * chunk_size
        end   = min((chunk+1) * chunk_size, sample_count)

        items.append((array[start:end], sr))

    return items

def print_results(writer, results, yamnet_classes, index, config):
    top, prediction = results
    seconds = index * float(config["seconds_per_chunk"])
    print(str(int(seconds // 60)) + ":" + str(int(seconds) % 60) + '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                    for i in top[0:1]))

    result = { "path" : "-", "seconds" : seconds }

    for i in top:
        result[yamnet_classes[i]] = float(prediction[i])

    writer.write(result)

def run_model_on_batch(yamnet, classes, params, pair):

    batch, sr = pair

    waveform = batch / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)

    # Predict YAMNet classes.
    scores, embeddings, spectrogram = yamnet(waveform)
    # Scores is a matrix of (time_frames, num_classes) classifier scores.
    # Average them along time to get an overall classifier output for the clip.
    prediction = np.mean(scores, axis=0)
    # Report the highest-scoring classes and their scores.
    top = np.argsort(prediction)[::-1][:5]

    return top, prediction

def setup_config(dictionary):
    return config.ConfigurationSet(
        config.config_from_env(prefix="MLCOMMONS"),
        config.config_from_yaml(config_path(), read_from_file=True),
        config.config_from_dict(dictionary),
    )

def config_path():
    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, ".mlcommons", "config.yaml")
    if os.path.exists(home_config_path):
        return home_config_path

    return os.path.join(os.path.dirname(__file__), "config", "default.yaml")

def setup_logging(arguments):

    logging_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    if arguments["verbose"]:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    elif arguments["verbose_info"]:
        logging.basicConfig(level=logging.INFO, format=logging_format)
    else:
        logging.basicConfig(level=logging.WARNING, format=logging_format)

    root_logger = logging.getLogger()

    if arguments["verbose"]:
        root_logger.setLevel(logging.DEBUG)
    elif arguments["verbose_info"]:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING)

    logging.getLogger("numba.core.ssa").setLevel(logging.CRITICAL)
    logging.getLogger("numba.core.byteflow").setLevel(logging.CRITICAL)
    logging.getLogger("numba.core.interpreter").setLevel(logging.CRITICAL)


if __name__ == '__main__':
    main()


