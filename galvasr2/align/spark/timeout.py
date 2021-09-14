import threading

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
  assert threading.current_thread() is threading.main_thread(), \
      "timeout() works only on the main thread"
  import signal

  class TimeoutError(Exception):
    pass

  def handler(signum, frame):
    raise TimeoutError()

  # set the timeout handler
  signal.signal(signal.SIGALRM, handler)
  signal.alarm(timeout_duration)
  try:
    result = func(*args, **kwargs)
  except TimeoutError as exc:
    result = default
  finally:
    signal.alarm(0)

  return result
