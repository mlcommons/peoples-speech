import mwxml
import pickle
from tqdm import tqdm
import requests
import os
from pyWikiCommons import pyWikiCommons


def get_dataset(xml_file ,output_file_name):
    dump = mwxml.Dump.from_file(open(xml_file))
    print(dump.site_info.name, dump.site_info.dbname)

    pages = []
    counter = 0
    for page in tqdm(dump):
        if page.namespace == 102:
            counter+=1
            for revision in page:
                revision = revision
            pages.append(revision)        
            print(counter)

    wikimedia = [i.to_json() for i in pages]

    import json
    with open(output_file_name , 'w') as file:
        json.dump(wikimedia, file)


# Esto solo sirve para .webm
def change_format(file_name):
    import subprocess
    changed_filename = os.rename(file_name, "temp.webm")
    print(true_link)
    subprocess.run(f'ffmpeg -i "temp.webm" {file_name}.mp3',shell=True)
    
def download_file(url,file_name):
    headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return file_name

def split_with_last_dot(string):
    for character in range(len(string)-1,0,-1):
        if string[character]==".":
            left = string[:character]
            right = string[character+1:]
            return [left,right] 


# Download, convert and upload files to GCP

import json
with open("wikimedia.json" , 'r') as file:
        previous_json = json.load(file)

counter = 0
database = {}
#for i in range(len(previous_json)):
for i in range(1):
    link = previous_json[i]["page"]["title"]
    dots = 0
    left,_ = split_with_last_dot(link)
    true_link, _ = split_with_last_dot(link)
    
    
    with open(link+".txt", 'w') as f:
        f.write(previous_json[i]["text"])
    
#    upload_and_delete_txt()
    
    if true_link in database.keys():
        database[true_link].append(link)
    else:
        database[true_link] = [link]
        try:
            url = pyWikiCommons.get_commons_url("File:" + true_link)
            download_file(url, true_link)
            change_format(true_link)
            
#            upload_and_delete_video()

            f = open("uploaded.txt", "a")
            f.write(f"{i},{true_link}\n")
            f.close()
        except:
            f = open("error.txt", "a")
            f.write(f"{i},{true_link}\n")
            f.close()
            