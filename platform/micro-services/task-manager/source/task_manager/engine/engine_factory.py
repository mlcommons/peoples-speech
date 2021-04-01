
from task_manager.engine.local_engine import LocalEngine

class EngineFactory:
    def __init__(self, config):
        self.config = config

    def create(self):
        if self.config["task_manager"]["engine"] == "LocalEngine":
            return LocalEngine(self.config)

        assert False, "Could not find engine " + self.config["task_manager"]["engine"]



