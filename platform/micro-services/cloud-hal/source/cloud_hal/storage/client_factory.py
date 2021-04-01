
from cloud_hal.storage.local_filesystem_client import LocalFilesystemClient

class ClientFactory:
    def __init__(self, name, config):
        self.name = name
        self.config = config

        if len(name) == 0:
            self.name = self.config["cloud_hal"]["storage"]["client"]

    def create(self):
        if self.name == "local":
            return LocalFilesystemClient(self.config)

        assert False, "Unknown client name " + self.name



