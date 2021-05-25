#!/bin/bash

set -euo pipefail

export SPARK_HOME=${SPARK_HOME:="$HOME/spark"}
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"
export SPARK_CONF_DIR="$SPARK_HOME/conf"

export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --allow-root --NotebookApp.open_browser=False --NotebookApp.ip='*' --NotebookApp.port=8880"

# https://stackoverflow.com/a/4774063
SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
TOP_PATH="$(readlink -f $SCRIPT_PATH/../..)"

# TODO: Ideally, this would be a bazeltarget rather than a standalone
# script. Right now, this will fail if the notebook tries to depend
# upon any shared object files. or arbitrary "data files".
# https://stackoverflow.com/a/16753536
export PYTHONPATH="$TOP_PATH:$TOP_PATH/galvasr2/align/:${PYTHONPATH:-}"

pyspark \
    --conf 'spark.executor.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true' \
    --conf 'spark.driver.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true' \
    --conf 'spark.hadoop.google.cloud.auth.service.account.enable=true' \
    "$@"
