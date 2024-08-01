import os
import time
import math
import numpy as np
from itertools import tee
from pynvml import *
import psutil
from datetime import datetime, timezone, timedelta

nvmlInit() # need initializztion here


def mean(x):
    if x == []:
        return 0.0
    return sum(x) / len(x)


def std(x):
    return np.std(x)


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def l2_distance(lon1, lat1, lon2, lat2):
    return math.sqrt( (lon2 - lon1) ** 2 + (lat2 - lat1) ** 2 )


# radian of the segment
def radian(lon1, lat1, lon2, lat2):
    dy = lat2 - lat1
    dx = lon2 - lon1
    r = 0.0

    if dx == 0:
        if dy >= 0:
            r = 1.5707963267948966 # math.pi / 2
        else: 
            r = 4.71238898038469 # math.pi * 1.5
        return round(r, 3)
    
    r = math.atan(dy / dx)
    # angle_in_degrees = math.degrees(angle_in_radians)
    if dx < 0:
        r = r + 3.141592653589793
    else:
        if dy < 0:
            r = r + 6.283185307179586
        else:
            pass
    return round(r, 3)

def degree(lon1, lat1, lon2, lat2):
    xdelta = lon2 - lon1
    ydelta = lat2 - lat1
    deg = math.degrees(math.atan2(ydelta, xdelta))
    return deg + 360 if deg < 0 else deg # [0~ 360]


def log_file_name():
    dt = datetime.now(timezone(timedelta(hours=8)))
    return dt.strftime("%Y%m%d_%H%M%S") + '.log'

# torch: number of parameters 
def num_of_model_params(models):
    n = 0
    if type(models) == list:
        for model in models:
            n += sum(p.numel() for p in model.parameters() if p.requires_grad) 
    else:
        n = sum(p.numel() for p in models.parameters() if p.requires_grad)
    return n


class Metrics:
    def __init__(self):
        self.dic = {} # {a:[], b:[], ...}
   
    def __str__(self) -> str:
        s = ''
        kvs = list(self.mean().items()) # + list(self.std().items())
        for i, (k, v) in enumerate(kvs):
            if i != 0:
                s += ','
            s = s + '{}={:.6f}'.format(k, v)
        return s

    def add(self, d_: dict):
        for k, v in d_.items():
            self.dic[k] = self.dic.get(k, []) + [v]
    
    def get(self, k):
        return self.dic.get(k, [])
    
    def mean(self, k = None):
        if type(k) == str:
            return mean(self.get(k)) # division by zero
        dic_mean = {}
        for k, v in self.dic.items():
            dic_mean[k] = mean(v)
        return dic_mean

    def std(self, k = None):
        if type(k) == str:
            return std(self.get(k)) # division by zero
        dic_mean = {}
        for k, v in self.dic.items():
            dic_mean[k] = std(v)
        return dic_mean
    

class Timer(object):
    _instance = None
    _last_t = 0.0

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self):
        self._last_t = time.time()

    def tick(self):
        _t = time.time()
        _delta = _t - self._last_t
        self._last_t = _t
        return _delta

timer = Timer()


class GPUInfo:
    _h = nvmlDeviceGetHandleByIndex(0)

    @classmethod
    def mem(cls):
        info = nvmlDeviceGetMemoryInfo(cls._h)
        return info.used // 1048576, info.total // 1048576 # in MB

class RAMInfo:
    @classmethod
    def mem(cls):
        return int(psutil.Process(os.getpid()).memory_info().rss / 1048576) # in MB
