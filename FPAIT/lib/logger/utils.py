import time, sys
import numpy as np

def time_for_file():
  ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
  return '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def time_string_short():
  ISOTIMEFORMAT='%Y%m%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def print_log(print_string, log):
  print("{}".format(print_string))
  if log is not None:
    log.write('{}\n'.format(print_string))
    log.flush()

def convert_secs2time(epoch_time, return_string=False):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  if return_string:
    return '{:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs)
  else:
    return need_hour, need_mins, need_secs

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __repr__(self):
    return ('{name}(val={val}, avg={avg}, count={count})'.format(name=self.__class__.__name__, **self.__dict__))
