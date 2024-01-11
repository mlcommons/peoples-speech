from requests.auth import HTTPBasicAuth
import os
import requests
from huggingface_hub import HfApi

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
HF_USER = ""
HF_TOKEN = ""
REPO_URL=""

pointer_directory = 'tar_pointers'
os.makedirs(pointer_directory, exist_ok=True)


def lfs_upload(repo_url, auth, fileobj, sha256, size, filename):
    lfs_url  = f"{repo_url}/info/lfs"
    batch_url = f"{lfs_url}/objects/batch"

    headers = {
        'Accept': 'application/vnd.git-lfs+json',
        'Content-Type': 'application/vnd.git-lfs+json',
    }
    j = {
        'operation': 'upload',
        'transfers': ['basic'],
        'ref': {'name': 'refs/heads/main'},
        'objects': [
            {'oid': sha256, 'size': size},
        ],
        'hash_algo': 'sha256',
    }
    r1 = requests.post(batch_url, headers=headers, auth=auth, json=j)
    try:
        r1.raise_for_status()
    except Exception:
        print(r1.content)
        raise
    j1 = r1.json()
    obj = j1['objects'][0]

    # already uploaded?
    if not 'actions' in obj:
        # print("Already uploaded") 
        return False

    upload_url = obj['actions']['upload']['href']
    r2 = requests.put(upload_url, headers={'Content-Type': 'application/octet-stream'}, data=fileobj)
    r2.raise_for_status()

    pointer = f'''version https://git-lfs.github.com/spec/v1
oid sha256:{sha256}
size {size}'''
    with open(f"tar_pointers/{filename}", "w") as p:
        p.write(pointer)
    return True

def lfs_upload_file(fileobj, filename, sha256):
    fileobj.seek(0, os.SEEK_SET)
    # TODO: workaround: requests was failing to upload our fileobj to s3
    data = fileobj.read()
    size = len(data)
    auth = HTTPBasicAuth(HF_USER, HF_TOKEN)
    repo_url = REPO_URL
    if size > 5*1024*1024*1024:
        blob_path = HfApi().upload_file(path_or_fileobj = data, path_in_repo = f"/audio/{filename}", repo_id = "remg1997/unsupervised_peoples_speech", repo_type = "dataset", token = HF_TOKEN)
        return True
    else:
        return lfs_upload(repo_url, auth, data, sha256, size, filename)
