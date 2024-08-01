import sys
sys.path.append('..')

from config import Config
from utils import tool_funcs


def remove_duplicate_points(src, min_length_tolerance = 0):
    if len(src) < min_length_tolerance:
        return src
    tgt = [v for i, v in enumerate(src) if i == 0 or v[0] != src[i-1][0]]
    return tgt


def enrich_spatial_features(src, space):
    # src = [length, 2]
    tgt = []
    lens = []
    degs = []
    for p1, p2 in tool_funcs.pairwise(src):
        lens.append(tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1]))
    for p1, p2 in tool_funcs.pairwise(src):
        degs.append(tool_funcs.degree(p1[0], p1[1], p2[0], p2[1]))

    xdiff = space['x_max'] - space['x_min']
    ydiff = space['y_max'] - space['y_min']
    
    for i in range(1, len(src) - 1):
        dist1 = lens[i-1] / Config.traj_distance_norm_denominator
        dist2 = lens[i] / Config.traj_distance_norm_denominator
        
        deg1 = degs[i-1] / 360
        deg2 = degs[i] / 360
        degdelta = abs(degs[i] - degs[i-1])
        degdelta = min(360 - degdelta, degdelta) / 180

        x = (src[i][0] - space['x_min']) / xdiff
        y = (src[i][1] - space['y_min']) / ydiff
        # tgt.append( [x, y, dist, radian] )
        tgt.append( [x, y, dist1, dist2, deg1, deg2, degdelta] )

    x = (src[0][0] - space['x_min']) / xdiff
    y = (src[0][1] - space['y_min']) / ydiff
    dist2 = lens[0] / Config.traj_distance_norm_denominator
    deg2 = degs[0] / 360
    tgt.insert(0, [x, y, 0.0, dist2, 0.0, deg2, 0.0])
    
    x = (src[-1][0] - space['x_min']) / xdiff
    y = (src[-1][1] - space['y_min']) / ydiff
    dist1 = lens[-1] / Config.traj_distance_norm_denominator
    deg1 = degs[-1] / 360
    tgt.append([x, y, dist1, 0.0, deg1, 0.0, 0.0])
    
    return tgt # [length, 7]


def enrich_spatial_features7(src):
    # src = [length, 2]
    tgt = []
    
    lens = []
    degs = []
    for p1, p2 in tool_funcs.pairwise(src):
        lens.append(tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1]))
    for p1, p2 in tool_funcs.pairwise(src):
        degs.append(tool_funcs.degree(p1[0], p1[1], p2[0], p2[1]))

    for i in range(1, len(src) - 1):
        dist1 = lens[i-1]
        dist2 = lens[i]
        
        deg1 = degs[i-1]
        deg2 = degs[i]
        degdelta = abs(degs[i] - degs[i-1])
        degdelta = min(360 - degdelta, degdelta)
        
        x = src[i][0] 
        y = src[i][1]
        tgt.append( [x, y, dist1, dist2, deg1, deg2, degdelta] )

    tgt.insert(0, [src[0][0], src[0][1], 0.0, lens[0], 0.0, degs[0], 0.0])
    
    tgt.append([src[-1][0], src[-1][1], lens[-1], 0.0, degs[-1], 0.0, 0.0])
    
    return tgt # [length, 7]


def padding_traj(lst):
    # pad the traj with the last point
    # lst: (bsz, seq, 4) list of list of list
    
    traj_len = list(map(len, lst))
    traj_max_len = max(traj_len)
    
    trajs_padded = []
    for t in lst:
        t_padded = t + [t[-1]] * (traj_max_len - len(t))
        trajs_padded.append(t_padded)
    
    return trajs_padded, traj_len
    
            
def preprocess_traj(src, space, min_length_tolerance = 0):
    src = remove_duplicate_points(src, min_length_tolerance)
    src = enrich_spatial_features(src, space)
    # src = enrich_spatial_features7(src)
    return src # 0-1 normalized


def traj_len(src):
    length = 0.0
    for p1, p2 in tool_funcs.pairwise(src):
        length += tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
    return length

