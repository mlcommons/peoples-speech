
import audiofile

def align_dataset(config, dataset):
    return [align_item(config, item) for item in dataset ]

def align_item(config, item):
    label, audio = item[0], item[1]

    duration_seconds = audiofile.duration(audio)

    if duration_seconds < float(config["forced_alignment"]["max_duration_seconds"]):
        return (label, audio) + item[2:]

    assert False, "Not implemented."

