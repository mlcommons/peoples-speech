from setuptools import setup, find_packages

from Cython.Build import cythonize
import numpy

print("GALVEZ:", find_packages())

setup(
    name="peoples-speech",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        # "git+https://github.com/apache/spark.git#egg=pyspark&subdirectory=python",
        "absl-py",
        "pandas",
        "matching",
        # "langid",
        "nemo_toolkit[asr]",
        "tqdm",
        "pydub",
        "ftfy",
        "srt",
        "pyarrow",
        "textdistance",
        "sox",
        "Cython",
        "webrtcvad",
        # From https://github.com/NVIDIA/NeMo/blob/main/tools/ctc_segmentation/requirements.txt
        # "ctc_segmentation==1.7.1",
        # "num2words",
        # End From
    ],
    include_package_data=True,
    package_data={'galvasr2': ['*.jar']},
    ext_modules=cythonize("galvasr2/align/smith_waterman.pyx"),
    include_dirs=[numpy.get_include()],
)
