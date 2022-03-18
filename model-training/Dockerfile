FROM nvcr.io/nvidia/pytorch:21.03-py3

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    libsndfile1 sox \
    libfreetype6 \
    python-setuptools swig \
    python-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install NeMo from fork
WORKDIR /workspace
RUN git clone https://github.com/JFCeron/NeMo --single-branch --branch the-peoples-speech --depth=1
RUN pip install -e NeMo

# Copy code files
WORKDIR /workspace/model-training
COPY . /workspace/model-training

# Uninstall stuff from base container
RUN pip uninstall -y sacrebleu torchtext torchvision

# Install dependencies
RUN pip install -r requirements.txt