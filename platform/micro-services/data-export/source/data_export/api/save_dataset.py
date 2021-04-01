
import csv

def save_dataset(arguments, dataset):
    with open(arguments["output_dataset_path"], "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        for item in dataset:
            writer.writerow(item)

