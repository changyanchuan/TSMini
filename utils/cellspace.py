import sys
sys.path.append('..')
import math

__all__ = ['merc2cell', 'create_cellspace']

def _xy2cellid(x, y, space):
    i_x = int(x - space['x_min']) // space['x_unit']
    i_y = int(y - space['y_min']) // space['y_unit']
    return i_x * space['y_size'] + i_y


def merc2cell(src, space):
    # convert and remove consecutive duplicates
    tgt = [_xy2cellid(*p, space) for p in src]
    # tgt = [v for i, v in enumerate(tgt) if i == 0 or v != tgt[i-1]]
    return tgt


def create_cellspace(x_min, x_max, y_min, y_max, x_unit, y_unit, buffer):
    x_min -= buffer
    x_max += buffer
    y_min -= buffer
    y_max += buffer
    x_size = int(math.ceil((x_max - x_min) / x_unit))
    y_size = int(math.ceil((y_max - y_min) / y_unit))
    size = x_size * y_size

    dic = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'x_unit': x_unit,
        'y_unit': y_unit,
        'buffer': buffer,
        'x_size': x_size,
        'y_size': y_size,
        'size': size,
    }
    return dic
