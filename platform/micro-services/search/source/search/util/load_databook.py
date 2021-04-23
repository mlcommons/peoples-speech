import csv

def load_databook(path):
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        return [row for row in reader]




