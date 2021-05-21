from flask import Flask, request
from flask_cors import CORS #comment this on deployment

import data_export

import logging

app = Flask(__name__)
CORS(app) #comment this on deployment

logger = logging.getLogger(__name__)

@app.route('/peoples_speech/export_dataset', methods=['GET', 'POST'])
def export_dataset():
    dataset = request.json["dataset"]
    output_dataset = request.json["output_dataset"]
    data_export.export_dataset(output_dataset, dataset)
    return { "results_path" : output_dataset }

