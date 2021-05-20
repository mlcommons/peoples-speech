#!/bin/bash

set -euo pipefail

export SPARK_HOME=${SPARK_HOME:="$HOME/spark"}
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"
export SPARK_CONF_DIR="$SPARK_HOME/conf"

export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --allow-root --NotebookApp.open_browser=False --NotebookApp.ip='*' --NotebookApp.port=8880"

pyspark \
    --conf 'spark.executor.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true' \
    --conf 'spark.driver.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true' \
    --conf 'spark.hadoop.google.cloud.auth.service.account.enable=true' \
    "$@"
