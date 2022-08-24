from concurrent.futures import ThreadPoolExecutor
import json
import requests
import threading
import urllib3

import internetarchive as ia
from tqdm import tqdm

LICENSE_WHITELIST = """(\
licenseurl:*creativecommons.org\/licenses\/by-sa\/4.0* OR \
licenseurl:*creativecommons.org\/licenses\/by-sa\/3.0* OR \
licenseurl:*creativecommons.org\/licenses\/by-sa\/2.5* OR \
licenseurl:*creativecommons.org\/licenses\/by-sa\/2.0* OR \
licenseurl:*creativecommons.org\/licenses\/by-sa\/1.0* OR \
licenseurl:*creativecommons.org\/publicdomain\/zero\/1.0* OR \
licenseurl:*creativecommons.org\/licenses\/by\/4.0* OR \
licenseurl:*creativecommons.org\/licenses\/by\/3.0* OR \
licenseurl:*creativecommons.org\/licenses\/by\/2.5* OR \
licenseurl:*creativecommons.org\/licenses\/by\/2.0* OR \
licenseurl:*creativecommons.org\/licenses\/by\/1.0* OR \
licenseurl:*creativecommons.org\/licenses\/publicdomain* OR \
licenseurl:*creativecommons.org\/publicdomain\/mark\/1.0* OR \
licenseurl:*usa.gov\/government-works* OR \
licenseurl:*creativecommons.org\/licenses\/cc0\/3.0* \
)"""  # pylint: disable=line-too-long,anomalous-backslash-in-string

QUERIES = dict(
    # TODO: Consider adding AND NOT format:ASR to disclude speech recognition-based labels.
    # ALL_CAPTIONED_DATA=f"{LICENSE_WHITELIST} AND (mediatype:audio OR mediatype:movies) AND (closed_captioning:yes OR format:SubRip OR format:\"Web Video Text Tracks\") AND (NOT access-restricted-item:TRUE)",
    # NON_CAPTIONED_DATA_WITH_TEXT=f"{LICENSE_WHITELIST} AND (format:DjVuTXT AND format:MP3 AND NOT format:SubRip) AND NOT (subject:'librivox')",
    # CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS=f'{LICENSE_WHITELIST} AND (mediatype:audio OR mediatype:movies) AND (closed_captioning:yes OR format:SubRip OR format:"Web Video Text Tracks") AND (NOT access-restricted-item:TRUE)',
    CC_BY_SA_ALL_AUDIO_LABELED_OR_UNLABELED=f'{LICENSE_WHITELIST} AND (mediatype:audio OR mediatype:movies)',
    # I used to have this, until I realized that I just needed to login (via 'ia configure') to download them.
    # AND (NOT access-restricted-item:TRUE)',
)

# default timeout is 12
REQUEST_KWARGS = {"timeout": 120.0}

def download_data(metadata_file, save_directory):
    def get_data(identifier):
        try:
            get_item_kwargs={"request_kwargs": REQUEST_KWARGS}
            ia.download(
                identifier,
                formats=[
                    # "SubRip",
                    "VBR MP3",
                    "MP3",
                    # "Web Video Text Tracks",
                    # "Closed Caption Text",
                ],
                destdir=save_directory,
                # Very important to set this. tf.io.gfile uses mtime in
                # nanoseconds, while archive.org uses mtime in seconds
                # (as far as I can tell). I could convert the
                # nanoseconds to seconds, of course, but don't want to
                # make an error.
                ignore_existing=True,
                # tf.io.gfile does not expose any functionality like os.utime
                no_change_timestamp=True,
                ignore_errors=True,
                silent=True,
                **get_item_kwargs,
            )
        except urllib3.exceptions.SSLError:
            print("ssl error for ", identifier)
            return
        except requests.exceptions.ConnectionError as err:
            print("Connection error for", identifier, ". Was:", err)
        except Exception as err:
            print("Unknown error for", identifier, ". Was:", err)

    ids = []
    with open(metadata_file, "rt") as fh:
        for line in fh:
            ids.append(json.loads(line)["identifier"])

    with ThreadPoolExecutor(10) as executor:
        list(tqdm(executor.map(get_data, ids), total=len(ids)))

def download_ids(query, id_save_file):
    print("Query IDs matching query")
    # default max_retries is 3
    search = ia.search_items(query, http_adapter_kwargs={"max_retries": 6})
    all_results = list(tqdm(search.iter_as_results()))
    with open(id_save_file, "wt") as fh:
        json.dump(all_results, fh)

def download_metadata(id_save_file, save_file):
    save_file_lock = threading.Lock()
    session = ia.get_session(http_adapter_kwargs={"max_retries": 15})
    with open(id_save_file, "rt") as fh:
        all_results = json.load(fh)
    try:
        with open(save_file, "rt") as fh:
            already_downloaded_ids = set(json.loads(line)["identifier"] for line in fh)
    except FileNotFoundError:
        already_downloaded_ids = set()
    with open(save_file, "at") as fh:
        def download_metadata_to_file(result: dict):
            if result["identifier"] in already_downloaded_ids:
                return
            item = ia.get_item(result["identifier"],
                               archive_session=session,
                               request_kwargs=REQUEST_KWARGS)
                
            metadata = item.item_metadata
            metadata["identifier"] = result["identifier"]
            with save_file_lock:
                json.dump(metadata, fh)
                fh.write("\n")

        with ThreadPoolExecutor(15) as executor:
            print("Download metadata")
            list(
                tqdm(executor.map(download_metadata_to_file, all_results), total=len(all_results))
            )


if __name__ == "__main__":
    for key, query in QUERIES.items():
        print(query)
        print(f"Dumping metadata for {key}")
        id_save_file = key + "_ids.jsonl"
        metadata_save_file = key + ".jsonl"
        # download_ids(query, id_save_file)
        # download_metadata(id_save_file, metadata_save_file)
        download_data(
            metadata_save_file,
            # Ryan: Change this output path
            f"gs://the-peoples-speech-west-europe/archive_org/unsupervised/June_2_2022/{key}"
            # f"download_output/{key}"
        )

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
