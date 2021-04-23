import csv
import os

def load_databook(path):
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        return [check_row(path, row) for row in reader]

def check_row(data_book_path, row):
    label, path = row

    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(data_book_path), path)

    assert os.path.exists(path),  "Path does not exist: " + path

    return label, path


