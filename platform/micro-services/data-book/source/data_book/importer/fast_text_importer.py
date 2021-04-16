
import cloud_hal.storage

from iso639 import languages

from urllib.request import urlopen
from smart_open import open
from gtts import gTTS

import os
import sys
import gzip
import shutil
import csv
import logging

logger = logging.getLogger(__name__)

logging.getLogger("gtts.tts").setLevel(logging.CRITICAL)
logging.getLogger("gtts.lang").setLevel(logging.CRITICAL)

class FastTextImporter:
    def __init__(self, config):
        self.config = config

        self.storage_client = cloud_hal.storage.client(config=config)

    def import_data_book(self):
        # download from FastText website to cache if needed
        embeddings_path = self.download_from_fast_text_to_cache()

        # load words
        words = self.load_words(embeddings_path)

        # generate audio from Google translate to cache if needed
        words_and_audio = self.add_audio_for_words(words)

        # write data book pages
        self.write_data_book_pages(words_and_audio)

    def download_from_fast_text_to_cache(self):
        language_code = self.get_language_code()

        cache_path = self.get_fasttext_cache_path(language_code)

        logger.debug("Checking for language " + language_code + " at " + cache_path)

        if not self.storage_client.exists(cache_path):
            logger.debug("It does not exist, downloading it to cache...")
            download_model(language_code, cache_path)

        return cache_path

    def get_language_code(self):
        language = languages.get(name=self.config["language"].capitalize())

        return language.alpha2

    def get_fasttext_cache_path(self, language_code):
        return os.path.join(self.storage_client.get_root(), "cache", "data_book", "fasttext", get_file_name(language_code))

    def load_words(self, embeddings_path):
        embeddings = load_vectors(embeddings_path, int(self.config["max_word_count"]))

        return embeddings.keys()

    def add_audio_for_words(self, words):
        audio_for_words = []

        for word in words:
            cache_path = self.get_word_cache_path(word)

            try:
                if not self.storage_client.exists(cache_path):
                    local_path = self.get_local_path_for_word(word)

                    tts = gTTS(word, lang=self.get_language_code())

                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    tts.save(local_path)

                    self.storage_client.copy(local_path, cache_path)

                logger.error("Generated audio for word: " + word)

                audio_for_words.append((word, cache_path))

            except Exception as e:
                logger.error("Failed to generate audio for word: " + word + " : " + str(e))

        return audio_for_words

    def get_word_cache_path(self, word):
        return os.path.join(self.storage_client.get_root(), "cache", "data_book", "fasttext", self.get_language_code(), word + ".mp3")

    def get_local_path_for_word(self, word):
        return os.path.join("/tmp", "cache", "data_book", "fasttext", self.get_language_code(), "local", word + ".mp3")

    def write_data_book_pages(self, words_and_audio):
        with open(self.config["output_path"], "w", newline="") as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"')
            for item in words_and_audio:
                writer.writerow(item)

def load_vectors(fname, max_word_count):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        if len(data) >= max_word_count:
            break

    logger.debug("Loaded " + str(len(data)) + " word embeddings.")

    return data

def _print_progress(downloaded_bytes, total_size):
    percent = float(downloaded_bytes) / total_size
    bar_size = 50
    bar = int(percent * bar_size)
    percent = round(percent * 100, 2)
    sys.stdout.write(" (%0.2f%%) [" % percent)
    sys.stdout.write("=" * bar)
    sys.stdout.write(">")
    sys.stdout.write(" " * (bar_size - bar))
    sys.stdout.write("]\r")
    sys.stdout.flush()

    if downloaded_bytes >= total_size:
        sys.stdout.write('\n')


def _download_file(url, write_file_name, chunk_size=2**13):
    print("Downloading %s" % url)
    response = urlopen(url)
    if hasattr(response, 'getheader'):
        file_size = int(response.getheader('Content-Length').strip())
    else:
        file_size = int(response.info().getheader('Content-Length').strip())
    downloaded = 0
    download_file_name = write_file_name + ".part"

    os.makedirs(os.path.dirname(download_file_name), exist_ok=True)

    with open(download_file_name, 'wb') as f:
        while True:
            chunk = response.read(chunk_size)
            downloaded += len(chunk)
            if not chunk:
                break
            f.write(chunk)
            _print_progress(downloaded, file_size)

    os.rename(download_file_name, write_file_name)

def _download_gz_model(gz_file_name, cache_file_name):

    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/%s" % gz_file_name
    _download_file(url, cache_file_name)

    return True


valid_lang_ids = {"af", "sq", "als", "am", "ar", "an", "hy", "as", "ast",
                  "az", "ba", "eu", "bar", "be", "bn", "bh", "bpy", "bs",
                  "br", "bg", "my", "ca", "ceb", "bcl", "ce", "zh", "cv",
                  "co", "hr", "cs", "da", "dv", "nl", "pa", "arz", "eml",
                  "en", "myv", "eo", "et", "hif", "fi", "fr", "gl", "ka",
                  "de", "gom", "el", "gu", "ht", "he", "mrj", "hi", "hu",
                  "is", "io", "ilo", "id", "ia", "ga", "it", "ja", "jv",
                  "kn", "pam", "kk", "km", "ky", "ko", "ku", "ckb", "la",
                  "lv", "li", "lt", "lmo", "nds", "lb", "mk", "mai", "mg",
                  "ms", "ml", "mt", "gv", "mr", "mzn", "mhr", "min", "xmf",
                  "mwl", "mn", "nah", "nap", "ne", "new", "frr", "nso",
                  "no", "nn", "oc", "or", "os", "pfl", "ps", "fa", "pms",
                  "pl", "pt", "qu", "ro", "rm", "ru", "sah", "sa", "sc",
                  "sco", "gd", "sr", "sh", "scn", "sd", "si", "sk", "sl",
                  "so", "azb", "es", "su", "sw", "sv", "tl", "tg", "ta",
                  "tt", "te", "th", "bo", "tr", "tk", "uk", "hsb", "ur",
                  "ug", "uz", "vec", "vi", "vo", "wa", "war", "cy", "vls",
                  "fy", "pnb", "yi", "yo", "diq", "zea"}

def download_model(lang_id, cache_file_name):
    """
        Download pre-trained common-crawl vectors from fastText's website
        https://fasttext.cc/docs/en/crawl-vectors.html
    """
    if lang_id not in valid_lang_ids:
        raise Exception("Invalid lang id. Please select among %s" %
                        repr(valid_lang_ids))

    _download_gz_model(get_file_name(lang_id), cache_file_name)

    return cache_file_name

def get_file_name(lang_id):
    file_name = "cc.%s.300.vec" % lang_id
    gz_file_name = "%s.gz" % file_name

    return gz_file_name


