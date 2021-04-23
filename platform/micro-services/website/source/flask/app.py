from flask import Flask
from flask_cors import CORS #comment this on deployment

from gcloud_sign_url import gcloud_sign_url

app = Flask(__name__)
CORS(app) #comment this on deployment

@app.route('/peoples_speech/train_url')
def get_train():
    config = {}
    return {'url' : gcloud_sign_url(config, "gs://the-peoples-speech-west-europe/peoples-speech-v0.7/train.tar.gz")}

@app.route('/peoples_speech/dev_url')
def get_dev():
    config = {}
    return {'url' : gcloud_sign_url(config, "gs://the-peoples-speech-west-europe/peoples-speech-v0.7/development.tar.gz")}

@app.route('/peoples_speech/test_url')
def get_test():
    config = {}
    return {'url' : gcloud_sign_url(config, "gs://the-peoples-speech-west-europe/peoples-speech-v0.7/test.tar.gz")}
