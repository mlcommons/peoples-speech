import hashlib
import logging
import multiprocessing as mp
import os
import tarfile
import tempfile
import time
import traceback
import typing

from tqdm import tqdm
import fire
import jsonlines

from lfs_upload import lfs_upload_file

log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)

def _hash_file(f: typing.BinaryIO) -> bytes:
    h = hashlib.sha256()
    while True:
        chunk = f.read(16384)
        h.update(chunk)
        if not chunk:
            break
    return h.digest()

def create_tar_file(args):
    index, filenames = args
    process_id = os.getpid()
    log_filename = os.path.join(log_directory, f'processing_log_{process_id}.txt')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    with tempfile.TemporaryFile() as f:
        tar_start = time.perf_counter()
        with tarfile.open(fileobj=f, mode='w') as tf:
            for path, arcpath in filenames:
                tf.add(path, arcpath)
        tar_duration = time.perf_counter() - tar_start
        sha256_start = time.perf_counter()
        f.seek(0, os.SEEK_SET)
        sha256_bytes = _hash_file(f)
        sha256_duration = time.perf_counter() - sha256_start

        upload_start = time.perf_counter()
        try:
            # cannot raise a requests error through multiprocess
            success = lfs_upload_file(f, f"{str(index).zfill(6)}.tar", sha256_bytes.hex())
        except Exception:
            traceback.print_exc()
            raise RuntimeError("failed to upload file")
        upload_duration = time.perf_counter() - upload_start

        logging.info(f'{index}.tar: archive={tar_duration:.2f}s, sha256={sha256_duration:.2f}s upload={upload_duration:.2f}s new_upload={success}')
        if not success:
            return None
        size = os.fstat(f.fileno()).st_size
        return size

# round a filesize up to its estimated size in a tar
# 512 byte header + file size rounded up to 512 bytes
def tar_entry_size(size: int) -> int:
    return 512 + (size + 0x1ff) & ~0x1ff

def iter_chunks(json_path, max_tar_size_bytes):
    chunk = []
    # 1024 bytes for tar end-of-archive entry
    size = 1024
    with jsonlines.open(json_path) as reader:
        for j in tqdm(reader, desc='json'):
            next_file_size = tar_entry_size(j['filesize'])
            if next_file_size >= max_tar_size_bytes:
                yield [(j['filepath'], j['arcpath'])]
                continue
                
            size += next_file_size
            if size >= max_tar_size_bytes:
                yield chunk
                chunk = []
                chunk.append((j['filepath'], j['arcpath'])) 
                size = 1024 + next_file_size
            else:
                chunk.append((j['filepath'], j['arcpath']))
    if chunk:
        yield chunk

def multiprocess_create_tar_files(json_lines_file, max_tar_size_gb, num_processes):
    max_tar_size_bytes = int(max_tar_size_gb * 1024 * 1024 * 1024)

    start_ts = time.perf_counter()
    processed_bytes = 0
    with mp.Pool(num_processes) as pool:
        chunks = list(enumerate(iter_chunks(json_lines_file, max_tar_size_bytes)))
        with tqdm(desc='upload', total=len(chunks)) as pbar:
            for size in pool.imap_unordered(create_tar_file, chunks):
                pbar.update(1)
                if size is not None:
                    elapsed = time.perf_counter() - start_ts
                    processed_bytes += size
                    bytes_s = processed_bytes / elapsed
                    gbytes_s = bytes_s / 1024 / 1024 / 1024
                    pbar.set_postfix({f"speed": f"{gbytes_s:.3f} GB/s"})

if __name__ == '__main__':
    fire.Fire(multiprocess_create_tar_files)
