from flask import Flask
from flask_cors import CORS #comment this on deployment

from data_export import export_dataset_by_id

app = Flask(__name__)
CORS(app) #comment this on deployment

@app.route('/peoples_speech/export_dataset/<dataset_id>')
def export_dataset(dataset_id):
    exported_id = export_dataset_by_id(dataset_id)
    return {"exported_id" : exported_id}

