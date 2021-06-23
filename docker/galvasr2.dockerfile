ARG cpu_base_image="nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Daniel Galvez"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN apt-get update && apt-get install -y --no-install-recommends \
        aria2 \
        build-essential \
        curl \
        dirmngr \
        emacs \
        ffmpeg \
        git \
        gpg-agent \
        less \
        libboost-all-dev \
        libeigen3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libbz2-dev \
        liblzma-dev \
        libpng-dev \
        libzmq3-dev \
        lsof \
        openjdk-11-jdk-headless \
        openjdk-11-dbg \
        pkg-config \
        rename \
        rsync \
        unzip \
        vim \
        wget \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN export CLOUDSDK_CORE_DISABLE_PROMPTS=1 CLOUDSDK_INSTALL_DIR=/install \
    && curl https://sdk.cloud.google.com | bash

# TODO: Set the configurations in launch_pyspark_notebook.sh here as well.
RUN /bin/bash -c "source /install/google-cloud-sdk/path.bash.inc && \
    curl -O https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz \
    && mkdir -p /install/spark \
    && tar zxf spark-3.1.2-bin-hadoop2.7.tgz -C /install/spark --strip-components=1 \
    && gsutil cp gs://hadoop-lib/gcs/gcs-connector-hadoop2-2.1.6.jar /install/spark/jars"

RUN echo "source /install/google-cloud-sdk/path.bash.inc" >> $HOME/.bashrc

ENV SPARK_HOME="/install/spark"
ENV PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"
ENV SPARK_CONF_DIR="$SPARK_HOME/conf"

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /install/miniconda3

COPY environment.yml /install/environment.yml
ENV PATH="/install/miniconda3/bin/:${PATH}"
RUN conda env create -f /install/environment.yml

RUN curl -L -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.7.4/bazelisk-linux-amd64 \
     && chmod a+x /usr/local/bin/bazel \
     && curl -L -o /usr/local/bin/ctpu https://dl.google.com/cloud_tpu/ctpu/latest/darwin/ctpu \
     && chmod a+x /usr/local/bin/ctpu

RUN conda init bash \
    && echo "conda init bash; conda activate 100k-hours-lingvo-3" >> $HOME/.bashrc

RUN curl -L -o /install/spark/jars/spark-tfrecord_2.12-0.3.0.jar \
    https://search.maven.org/remotecontent?filepath=com/linkedin/sparktfrecord/spark-tfrecord_2.12/0.3.0/spark-tfrecord_2.12-0.3.0.jar

RUN mkdir -p /install/mad/ \
    && curl -L -o /install/mad/mad.tar.gz https://downloads.sourceforge.net/project/mad/libmad/0.15.1b/libmad-0.15.1b.tar.gz \
    && cd /install/mad \
    && tar zxf mad.tar.gz --strip-components=1 \
    && sed -i '/-fforce-mem/d' ./configure \
    && ./configure --prefix=/usr --disable-debugging --enable-fpm=64bit \
    && make -j $(nproc) install

RUN mkdir -p /install/lame/ \
    && curl -L -o /install/lame/lame.tar.gz https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz \
    && cd /install/lame \
    && tar zxf lame.tar.gz --strip-components=1 \
    && ./configure --prefix=/usr \
    && make -j $(nproc) install

RUN mkdir -p /install/flac/ \
    && curl -L -o /install/flac/flac.tar.xz https://downloads.xiph.org/releases/flac/flac-1.3.3.tar.xz \
    && cd /install/flac \
    && tar xf flac.tar.xz --strip-components=1 \
    && ./configure --prefix=/usr --disable-dependency-tracking --disable-debug --enable-static \
    && make -j $(nproc) install

RUN mkdir -p /install/sox/ \
    && curl -L -o /install/sox/sox.tar.gz https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz \
    && cd /install/sox \
    && tar zxf sox.tar.gz --strip-components=1 \
    && ./configure --prefix=/usr \
    && make -j $(nproc) install

RUN cd /install/spark/python && conda run -n 100k-hours-lingvo-3  python setup.py install

RUN    echo 'spark.eventLog.enabled  true' >> $SPARK_CONF_DIR/spark-defaults.conf
# \
#     && echo 'spark.eventLog.dir file:///development/lingvo-source/spark-events' >> $SPARK_CONF_DIR/spark-defaults.conf \
#     && echo 'spark.history.fs.logDirectory file:///development/lingvo-source/spark-events' >> $SPARK_CONF_DIR/spark-defaults.conf

RUN apt-get update && apt-get install -y --no-install-recommends automake autoconf gfortran libtool subversion

RUN export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` \
    && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install -y gcsfuse

    # && apt-get update \
    # && apt-get install -y kmod \
    # && modprobe fuse \
    # && gcsfuse --implicit-dirs the-peoples-speech-west-europe \
    #    $HOME/the-peoples-speech-west-europe-bucket
# gcsfuse --implicit-dirs the-peoples-speech-west-europe $HOME/the-peoples-speech-west-europe-bucket
RUN mkdir -p /spark-events \
    && mkdir -p $HOME/the-peoples-speech-west-europe-bucket

COPY third_party/kaldi/tools /opt/kaldi/tools
RUN cd /opt/kaldi/tools \
    && extras/install_openblas.sh \
    && make -j $(nproc)

COPY third_party/kaldi/egs /opt/kaldi/egs
COPY third_party/kaldi/src /opt/kaldi/src
RUN cd /opt/kaldi/src \
    && ./configure --use-cuda=yes --cudatk-dir=/usr/local/cuda --mathlib=OPENBLAS --cuda-arch="-gencode arch=compute_75,code=sm_75" \
    && make -j $(nproc) depend && make -j $(nproc)

RUN cd /opt/kaldi/egs/aspire/s5 \
    && wget http://kaldi-asr.org/models/1/0001_aspire_chain_model_with_hclg.tar.bz2 \
    && tar jxfv 0001_aspire_chain_model_with_hclg.tar.bz2 \
    && steps/online/nnet3/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
       data/lang_chain exp/nnet3/extractor exp/chain/tdnn_7b exp/tdnn_7b_chain_online

# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888
EXPOSE 8880

WORKDIR "/development/lingvo-source"

CMD ["/bin/bash", "-c"]
