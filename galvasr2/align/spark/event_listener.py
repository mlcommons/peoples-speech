# Adapter from second half of this answer: https://stackoverflow.com/a/44084038

class SparkListener:
  def onApplicationEnd(self, applicationEnd):
    pass
  def onApplicationStart(self, applicationStart):
    pass
  def onBlockManagerRemoved(self, blockManagerRemoved):
    pass
  def onBlockUpdated(self, blockUpdated):
    pass
  def onEnvironmentUpdate(self, environmentUpdate):
    pass
  def onExecutorAdded(self, executorAdded):
    pass
  def onExecutorMetricsUpdate(self, executorMetricsUpdate):
    pass
  def onExecutorRemoved(self, executorRemoved):
    pass
  def onJobEnd(self, jobEnd):
    pass
  def onJobStart(self, jobStart):
    pass
  def onOtherEvent(self, event):
    pass
  def onStageCompleted(self, stageCompleted):
    pass
  def onStageSubmitted(self, stageSubmitted):
    pass
  def onTaskEnd(self, taskEnd):
    pass
  def onTaskGettingResult(self, taskGettingResult):
    pass
  def onTaskStart(self, taskStart):
    pass
  def onUnpersistRDD(self, unpersistRDD):
    pass
  def onBlockManagerAdded(self, _):
    pass
  class Java:
    implements = ["org.apache.spark.scheduler.SparkListenerInterface"]

class WriteTaskEndListener(SparkListener):
  def __init__(self):
    self._value = 0

  def onJobStart(self, _jobEnd):
    self._value = 0

  def onTaskEnd(self, taskEnd):
    self._value += taskEnd.taskMetrics().outputMetrics().recordsWritten()

  @property
  def value(self):
    return self._value
