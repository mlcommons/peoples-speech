import os

def is_path(path):
    if os.path.exists(path):
        return True

    if path.startswith("s3://"):
        return True

    if path.startswith("gs://"):
        return True

    return False

