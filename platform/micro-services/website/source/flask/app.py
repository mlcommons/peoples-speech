from flask import Flask, request
from flask_cors import CORS #comment this on deployment
import json
import os

from gcloud_sign_url import gcloud_sign_url

app = Flask(__name__)
CORS(app) #comment this on deployment

@app.route('/peoples_speech/train_url', methods=['GET', 'POST'])
def get_train():
    config = load_config()
    if not request.json["email"] in config["peoples_speech"]["access_list"]:
        return {"url" : "not_permitted"}
    return {'url' : gcloud_sign_url(config, "gs://the-peoples-speech-west-europe/peoples-speech-v0.7/train.tar.gz")}

@app.route('/peoples_speech/dev_url', methods=['GET', 'POST'])
def get_dev():
    config = load_config()
    if not request.json["email"] in config["peoples_speech"]["access_list"]:
        return {"url" : "not_permitted"}
    return {'url' : gcloud_sign_url(config, "gs://the-peoples-speech-west-europe/peoples-speech-v0.7/development.tar.gz")}

@app.route('/peoples_speech/test_url', methods=['GET', 'POST'])
def get_test():
    config = load_config()
    if not request.json["email"] in config["peoples_speech"]["access_list"]:
        return {"url" : "not_permitted"}
    return {'url' : gcloud_sign_url(config, "gs://the-peoples-speech-west-europe/peoples-speech-v0.7/test.tar.gz")}

def load_config():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "default.json")
    with open(path) as config_file:
        return json.load(config_file)

