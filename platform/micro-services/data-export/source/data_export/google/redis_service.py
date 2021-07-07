
import subprocess
import logging

logger = logging.getLogger(__name__)

class RedisService:
    def __init__(self, config):
        self.config = config

        self.startup()

    def startup(self):
        self.instances = self.get_instances()

        if not "exporter-work-queue" in self.instances:
            self.start_instance()

    def start_instance(self):
        command = ["gcloud", "redis", "instances", "create", "exporter-work-queue", "--size=2", "--region", "europe-west4"]

        subprocess.run(command)

    def get_instances(self):
        command = ["gcloud", "redis", "instances", "describe", "exporter-work-queue", "--region", "europe-west4"]

        process = subprocess.run(command, capture_output=True)

        stdout = process.stdout.decode()

        logger.debug("stdout: " + stdout)

        # parse output
        instances = {}

        for line in stdout.split('\n'):
            logger.debug("parsing line: '" + line + "'")
            if line.find("host:") == 0:
                host = line[5:].strip()

            if line.find("name:") == 0:
                name = line[5:].strip().split("/")[-1]
                instances[name] = host

        return instances

    def kwargs(self):
        return { "host" : self.instances["exporter-work-queue"], "port" : 6379, "db" : 0 }



