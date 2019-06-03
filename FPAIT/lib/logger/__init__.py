try:
  from .logger import Logger
except ImportError:
  class Logger(object):
    def __init__(self, log_dir):
      self.log_dir = log_dir
    def __repr__(self):
      return ('{name}-None: (dir={log_dir})'.format(name=self.__class__.__name__, **self.__dict__))

from .utils import time_for_file, time_string, print_log
from .utils import AverageMeter
from .utils import convert_secs2time
