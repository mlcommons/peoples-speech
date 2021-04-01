

class LocalEngine:
    def __init__(self, config):
        self.config = config

    def run(self, function, *argv):
        return function(*argv)



