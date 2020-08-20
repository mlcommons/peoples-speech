from concurrent.futures import ThreadPoolExecutor
import gzip
import json

import internetarchive as ia
from tqdm import tqdm

LICENSE_WHITELIST = "(licenseurl:*creativecommons.org\/publicdomain\/zero\/1.0* OR licenseurl:*creativecommons.org\/licenses\/by\/4.0* OR licenseurl:*creativecommons.org\/licenses\/by\/3.0* OR licenseurl:*creativecommons.org\/licenses\/by\/2.0* OR licenseurl:*creativecommons.org\/licenses\/by\/1.0*)" #pylint: disable=line-too-long,anomalous-backslash-in-string

QUERIES = dict(
#  CAPTIONED_DATA = f"{LICENSE_WHITELIST} and format:ASR",
  AUDIO = f"{LICENSE_WHITELIST} and mediatype:audio",
  MOVIES = f"{LICENSE_WHITELIST} and mediatype:movies"
)


def download_data(metadata_file, save_directory):
  def get_data(identifier):
    ia.download(identifier,
                formats=["ASR", "SubRip", "MP3"], #, "Ogg Video"],
                destdir=save_directory,
                # Very import to set this. tf.io.gfile uses mtime in
                # nanoseconds, while archive.org uses mtime in seconds
                # (as far as I can tell). I could convert the
                # nanoseconds to seconds, of course, but don't want to
                # make an error.
                ignore_existing=True,
                # tf.io.gfile does not expose any functionality like os.utime
                no_change_timestamp=True,
                ignore_errors=True)

  ids = []
  with gzip.open(metadata_file, "rt") as fh:
    for line in fh:
      ids.append(json.loads(line)["identifier"])

  with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(get_data, ids), total=len(ids)))


def download_metadata(query, save_file):
  search = ia.search_items(query)
  all_results = list(tqdm(search.iter_as_results()))

  def get_metadata(result: dict):
    item = ia.get_item(result["identifier"],
                           archive_session=search.session)
    metadata = item.item_metadata
    metadata["identifier"] = result["identifier"]
    return metadata

  with ThreadPoolExecutor() as executor:
    metadata_list = list(tqdm(executor.map(get_metadata, all_results), total=len(all_results)))
  with gzip.open(save_file, "wt") as fh:
    for metadata in metadata_list:
      json.dump(metadata, fh)
      fh.write("\n")

if __name__ == '__main__':
  # for key, query in QUERIES.items():
  #   print(f"Dumping metadata for {key}")
  #   save_file = key + ".jsonl.gz"
  #   download_metadata(query, save_file)

  # download_data("CAPTIONED_DATA.jsonl.gz", "gs://the-peoples-speech-west-europe/archive_org/Aug_18_2020/CAPTIONED_DATA")
  download_data("MOVIES.jsonl.gz", "gs://the-peoples-speech-west-europe/archive_org/Aug_19_2020/MOVIES")
  download_data("AUDIO.jsonl.gz", "gs://the-peoples-speech-west-europe/archive_org/Aug_19_2020/AUDIO")
  # download_data("one_line.jsonl", "CAPTIONED_DATA")
