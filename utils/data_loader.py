import os
import time
import logging
import pickle
import h5py
import math
import random
from torch.utils.data import Dataset


# Load traj dataset for trajsimi learning
def read_trajsimi_traj_dataset(file_path):
    logging.info('[trajsimi traj dataset] Start loading.')
    _time = time.time()

    file_path += '.h5'
    datafile = h5py.File(file_path, 'r')
    merc_range = datafile.attrs['merc_range'][()].tolist()
    
    offset_idx = int(datafile.attrs['n'] * 0.7) # use eval dataset
    l = 10000
    trajs = []
    for i in range(l):
        idx = i + offset_idx
        trajs.append(datafile['/trajs_merc/%s' % idx][:].tolist())
    
    datafile.close()
    
    train_idx = (int(l*0), int(l*0.7))
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))
    trains = trajs[train_idx[0] : train_idx[1]]
    evals = trajs[eval_idx[0] : eval_idx[1]]
    tests = trajs[test_idx[0] : test_idx[1]]

    logging.info("[trajsimi traj dataset] Loaded. @={:.2f}. traj: #total={} (trains/evals/tests={}/{}/{})" \
                .format(time.time() - _time, l, len(trains), len(evals), len(tests)))
    return trains, evals, tests, merc_range


# Load simi dataset for trajsimi learning
def read_trajsimi_simi_dataset(file_path):
    logging.info('[trajsimi simi dataset] Start loading.')
    _time = time.time()
    if not os.path.exists(file_path):
        logging.error('[trajsimi simi dataset does not exist')
        exit(200)

    with open(file_path, 'rb') as fh:
        trains_simi, evals_simi, tests_simi, max_distance = pickle.load(fh)
        logging.info("[trajsimi simi dataset] Loaded. @={:.2f}, trains/evals/tests={}/{}/{}" \
                .format(time.time() - _time, len(trains_simi), len(evals_simi), len(tests_simi)))
        return trains_simi, evals_simi, tests_simi, max_distance


class TrajSimiDatasetTraining(Dataset):
    def __init__(self, trains_traj, batchsize):
        self.trains_traj = trains_traj
        self.n = len(self.trains_traj)
        
        self.batchsize = batchsize
        self.niters = math.ceil( (self.n / batchsize)**2 )
    
    def __getitem__(self, index):
        sampled_idxs = random.sample(range(self.n), k = self.batchsize)
        trajs = [self.trains_traj[d_idx] for d_idx in sampled_idxs]

        return trajs, sampled_idxs
    
    def __len__(self):
        return self.niters

