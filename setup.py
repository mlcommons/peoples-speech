from setuptools import setup, find_packages

from Cython.Build import cythonize
import numpy

print("GALVEZ:", find_packages())

setup(
    name="peoples-speech",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "pyspark==3.1.2",
        "absl-py",
        "pandas",
        "matching",
        "langid",
        "tqdm",
        "pydub",
        "ftfy",
        "srt",
        "pyarrow",
        "textdistance",
        "sox",
        "Cython",
        "webrtcvad"
    ],
    include_package_data=True,
    package_data={'galvasr2': ['*.jar']},
    ext_modules=cythonize("galvasr2/align/smith_waterman.pyx"),
    include_dirs=[numpy.get_include()],
)
