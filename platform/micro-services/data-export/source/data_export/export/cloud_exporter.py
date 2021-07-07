
import requests

import logging

logger = logging.getLogger()

class CloudExporter:
    def __init__(self, config):
        self.config = config

    def export(self, path, dataset):
        url = self.config["exporter"]["endpoint"] + ":" + self.get_port() + "/peoples_speech/export_dataset"
        data = {
            "dataset" : dataset,
            "output_dataset" : path
        }

        logger.debug("Submitting POST request to url " + str(url))
        logger.debug(" with data " + str(data))
        response = requests.post(url, json=data)

        if response.status_code == 200:
            print("Success")
            print(response.json())
        else:
            print("Failed to submit with error: " + str(response.status_code))

    def get_port(self):
        return "5000"

