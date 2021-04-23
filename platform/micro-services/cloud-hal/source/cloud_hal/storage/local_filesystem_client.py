
import os
import shutil

class LocalFilesystemClient:
    def __init__(self, config):
        self.config = config

        self.root = "/tmp"

        if "cloud_hal" in config:
            if "storage" in config["cloud_hal"]:
                if "root" in config["cloud_hal"]["storage"]:
                    self.root = config["cloud_hal"]["storage"]["root"]

    def exists(self, path):
        return os.path.exists(path)

    def get_root(self):
        return self.root

    def copy(self, src, dst):
        shutil.copyfile(src, dst)


