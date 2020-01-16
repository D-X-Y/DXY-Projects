# https://sherlockliao.github.io/2017/06/29/tensorboard/
from pathlib import Path
import time, sys, numpy as np

class Logger(object):
  
  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    self.log_dir   = Path(log_dir)
    self.model_dir = Path(log_dir) / 'checkpoint'
    self.log_dir.mkdir  (parents=True, exist_ok=True)
    self.model_dir.mkdir(parents=True, exist_ok=True)
    self.logger_path = self.log_dir / '{:}.log'.format( time.strftime('%d-%h-at-%H-%M-%S', time.gmtime(time.time())) )
    self.logger_file = open(self.logger_path, 'w')

  def __repr__(self):
    return ('{name}(dir={log_dir})'.format(name=self.__class__.__name__, **self.__dict__))

  def write(self, xstr):
    self.logger_file.write(xstr)
  
  def flush(self):
    self.logger_file.flush()
  
  def close(self):
    self.logger_file.close()
