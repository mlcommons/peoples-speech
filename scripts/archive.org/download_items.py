from concurrent.futures import ThreadPoolExecutor
import gzip
import json

import internetarchive as ia
from tqdm import tqdm

LICENSE_WHITELIST = "(licenseurl:*creativecommons.org\/publicdomain\/zero\/1.0* OR licenseurl:*creativecommons.org\/licenses\/by\/4.0* OR licenseurl:*creativecommons.org\/licenses\/by\/3.0* OR licenseurl:*creativecommons.org\/licenses\/by\/2.5* OR licenseurl:*creativecommons.org\/licenses\/by\/2.0* OR licenseurl:*creativecommons.org\/licenses\/by\/1.0* OR licenseurl:*creativecommons.org\/licenses\/publicdomain* OR licenseurl:*creativecommons.org\/publicdomain\/mark\/1.0* OR licenseurl:*usa.gov\/government-works* OR licenseurl:*creativecommons.org\/licenses\/cc0\/3.0*)"  # pylint: disable=line-too-long,anomalous-backslash-in-string

QUERIES = dict(
  # TODO: Consider adding AND NOT format:ASR to disclude speech recognition-based labels.
  #ALL_CAPTIONED_DATA=f"{LICENSE_WHITELIST} AND (mediatype:audio OR mediatype:movies) AND (closed_captioning:yes OR format:SubRip OR format:\"Web Video Text Tracks\") AND (NOT access-restricted-item:TRUE)",
  # NON_CAPTIONED_DATA_WITH_TEXT=f"{LICENSE_WHITELIST} AND (format:DjVuTXT AND format:MP3 AND NOT format:SubRip) AND NOT (subject:'librivox')",
  EXPANDED_LICENSES_FILTERED_ACCESS=f"{LICENSE_WHITELIST} AND (mediatype:audio OR mediatype:movies) AND (closed_captioning:yes OR format:SubRip OR format:\"Web Video Text Tracks\") AND (NOT access-restricted-item:TRUE)",
)


def download_data(metadata_file, save_directory):
  def get_data(identifier):
    ia.download(identifier,
                formats=["SubRip", "MP3", "Web Video Text Tracks", "Closed Caption Text"],
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

  with ThreadPoolExecutor(15) as executor:
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

  with ThreadPoolExecutor(15) as executor:
    metadata_list = list(tqdm(executor.map(get_metadata, all_results), total=len(all_results)))
  with gzip.open(save_file, "wt") as fh:
    for metadata in metadata_list:
      json.dump(metadata, fh)
      fh.write("\n")

if __name__ == '__main__':
  for key, query in QUERIES.items():
    print(query)
    print(f"Dumping metadata for {key}")
    save_file = key + ".jsonl.gz"
    download_metadata(query, save_file)
    download_data(save_file, f"gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/{key}")

  # download_data("CAPTIONED_DATA.jsonl.gz", "gs://the-peoples-speech-west-europe/archive_org/Aug_18_2020/CAPTIONED_DATA")
  # download_data("MOVIES.jsonl.gz", "gs://the-peoples-speech-west-europe/archive_org/Aug_19_2020/MOVIES")
  # download_data("AUDIO.jsonl.gz", "gs://the-peoples-speech-west-europe/archive_org/Aug_19_2020/AUDIO")
  # download_data("one_line.jsonl", "CAPTIONED_DATA")
  # download_data("ALL_CAPTIONED_DATA.jsonl.gz", "gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA")

# https://archive.org/services/docs/api/metadata-schema/index.html#mediatype
# usage notes:
# texts: books, articles, newspapers, magazines, any documents with content that contains text
# etree: live music concerts, items should only be uploaded for artists with collections in the etree “Live Music Archive” community
# audio: any item where the main media content is audio files, like FLAC, mp3, WAV, etc.
# movies: any item where the main media content is video files, like mpeg, mov, avi, etc.
# software: any item where the main media content is software intended to be run on a computer or related device such as gaming devices, phones, etc.
# image: any item where the main media content is image files (but is not a book or other text item), like jpeg, gif, tiff, etc.
# data: any item where the main content is not media or web pages, such as data sets
# web: any item where the main content is copies of web pages, usually stored in WARC or ARC format
# collection: designates the item as a collection that can “contain” other items
# account: designates the item as being a user account page, can only be set by internal archive systems
