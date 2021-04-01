
from cloud_hal.storage.client_factory import ClientFactory

def client(name="local", config={}):
    return ClientFactory(name, config).create()



