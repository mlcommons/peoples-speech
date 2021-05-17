import subprocess
import shlex
import tqdm

def main():
    total_duration = 0.0
    missing_files = []
    with open("/development/lingvo-source/paths.txt/part-00000-733bb7f7-f80a-4e25-b2e0-f7c48b106103-c000.txt") as fh:
        lines = fh.readlines()
        for line in tqdm.tqdm(lines):
            line = line.rstrip("\n")
            try:
                try:
                    duration = subprocess.check_output(shlex.split(f"soxi -D \"{line}\""))
                except ValueError:
                    print("GALVEZ:", line)
                    raise
                duration = duration.rstrip(b"\n")
                duration = float(duration)
            except subprocess.CalledProcessError:
                missing_files.append(line)
                duration = 0.0
            total_duration += duration
    print(missing_files)
    print(total_duration)

if __name__ == '__main__':
    main()
