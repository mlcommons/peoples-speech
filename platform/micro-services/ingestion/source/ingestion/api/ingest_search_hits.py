
from cloud_hal.api import is_path

def ingest_search_hits(config, hits):
    dataset = []

    for hit in hits:
        label, audio_path = hit

        audio_path = ingest_audio(audio_path)

        dataset.append((label, audio_path))

    return dataset

def ingest_audio(audio_path):
    if is_path(audio_path):
        return audio_path

    assert False, "Not implemented for " + audio_path

