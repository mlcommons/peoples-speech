# People's Speech Data Pipelines

Installation


```
sudo apt-get install git-lfs
# Set up a virtual environment of some sort
pip install numpy Cython
python setup.py develop
cp galvasr2/*.jar $(python -c "import pyspark; print(pyspark.__path__[0])")/jars
```

Run forced alignment pipeline.

```
python galvasr2/align/spark/align_cuda_decoder.py --stage=0
```

