ARG cpu_base_image="ubuntu:18.04"
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
    curl -O https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz \
    && mkdir -p /install/spark \
    && tar zxf spark-3.0.0-bin-hadoop2.7.tgz -C /install/spark --strip-components=1 \
    && gsutil cp gs://hadoop-lib/gcs/gcs-connector-hadoop2-2.1.6.jar /install/spark/jars"

ENV SPARK_HOME="/install/spark"
ENV PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"
ENV SPARK_CONF_DIR="$SPARK_HOME/conf"

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /install/miniconda3

COPY environment2.yml /install/environment2.yml
ENV PATH="/install/miniconda3/bin/:${PATH}"
RUN conda env create -f /install/environment2.yml

RUN curl -L -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.7.4/bazelisk-linux-amd64 \
     && chmod a+x /usr/local/bin/bazel \
     && curl -L -o /usr/local/bin/ctpu https://dl.google.com/cloud_tpu/ctpu/latest/darwin/ctpu \
     && chmod a+x /usr/local/bin/ctpu

RUN conda init bash \
    && echo "conda init bash; conda activate 100k-hours-lingvo-3" >> $HOME/.bashrc

RUN curl -L -o /install/spark/jars/spark-tfrecord_2.12-0.3.0.jar \
    https://search.maven.org/remotecontent?filepath=com/linkedin/sparktfrecord/spark-tfrecord_2.12/0.3.0/spark-tfrecord_2.12-0.3.0.jar

RUN curl -L -O https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1-Linux-x86_64.sh \
    && chmod +x cmake-3.19.1-Linux-x86_64.sh \
    && ./cmake-3.19.1-Linux-x86_64.sh --exclude-subdir --skip-license --prefix=/usr/local/

# https://100k-hours.slack.com/archives/CLQAYP797/p1605036830303800
# https://100k-hours.slack.com/archives/CLQAYP797/p1605045990306700?thread_ts=1605039855.304800&cid=CLQAYP797
# TODO: Download a particular version of kenlm, not just the latest one.
RUN mkdir -p /install/kenlm/ \
    && curl -L -o /install/kenlm/kenlm.tar.gz https://kheafield.com/code/kenlm.tar.gz \
    && cd /install/kenlm/ \
    && tar zxf kenlm.tar.gz --strip-components=1 \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make -j $(nproc)

# This is probably not necessary. I initially believed I needed to run autoconf, but this is no longer the case.
# RUN apt-get update && apt-get install -y --no-install-recommends autoconf autotools-dev automake libtool

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

RUN mkdir -p /install/sox/ \
    && curl -L -o /install/sox/sox.tar.gz https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz \
    && cd /install/sox \
    && tar zxf sox.tar.gz --strip-components=1 \
    && ./configure --prefix=/usr \
    && make -j $(nproc) install

# RUN apt-get install libdw1 libpci3 libslang2 libunwind8 linux-tools-common \
#     && wget http://launchpadlibrarian.net/301849399/linux-tools-4.9.0-12_4.9.0-12.13_amd64.deb \
#     && dpkg -i linux-tools-4.9.0-12_4.9.0-12.13_amd64.deb \
#     && wget http://launchpadlibrarian.net/301849369/linux-tools-4.9.0-12-generic_4.9.0-12.13_amd64.deb \
#     && dpkg -i linux-tools-4.9.0-12-generic_4.9.0-12.13_amd64.deb

# --cap-add SYS_ADMIN

# RUN apt-get update && apt-get install -y --no-install-recommends bison flex gcc-6 

# RUN git clone --single-branch --branch v4.9 https://github.com/torvalds/linux.git /install/linux \
#     && CC=gcc-6 make -C /install/linux/tools/perf install

RUN cd /install/spark/python && conda run -n 100k-hours-lingvo-3  python setup.py install

# RUN    echo 'spark.driver.extraJavaOptions="-Dio.netty.tryReflectionSetAccessible=true"' >> $SPARK_CONF_DIR/spark-defaults.conf \
#     && echo 'spark.driver.memory=8g' >> $SPARK_CONF_DIR/spark-defaults.conf \
#     && echo 'spark.executor.extraJavaOptions="-Dio.netty.tryReflectionSetAccessible=true"' >> $SPARK_CONF_DIR/spark-defaults.conf \
#     && echo 'spark.executor.memory=8g' >> $SPARK_CONF_DIR/spark-defaults.conf

RUN    echo 'spark.eventLog.enabled  true' >> $SPARK_CONF_DIR/spark-defaults.conf \
    && echo 'spark.eventLog.dir file:///development/lingvo-source/spark-events' >> $SPARK_CONF_DIR/spark-defaults.conf \
    && echo 'spark.history.fs.logDirectory file:///development/lingvo-source/spark-events' >> $SPARK_CONF_DIR/spark-defaults.conf

# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888
EXPOSE 8880

WORKDIR "/development/lingvo-source"

CMD ["/bin/bash", "-c"]
