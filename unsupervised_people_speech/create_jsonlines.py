import os
import time

import fire
import jsonlines

def get_file_info(root_dir: str, output_file: str):
    start_time = time.time()
    with jsonlines.open(f'{output_file}', 'w') as writer:
        for root, dirs, files in os.walk(root_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                filesize = os.path.getsize(filepath)
                _, file_extension = os.path.splitext(filename)
                arcpath = os.path.relpath(filepath, root_dir)
                writer.write({
                    'filepath': filepath,
                    'arcpath': arcpath,
                    'filesize': filesize,
                    'file_extension': file_extension
                })
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")

if __name__ == '__main__':
    fire.Fire(get_file_info)
