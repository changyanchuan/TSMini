import logging
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
from functools import partial

from model.lambdaloss import lambdaLoss
from config import Config as Config
from utils import tool_funcs
from utils.data_loader import read_trajsimi_simi_dataset, read_trajsimi_traj_dataset, TrajSimiDatasetTraining
from utils.traj import *
from utils.cellspace import *
from model.tsmini import TSMini


class TrajSimi:
    def __init__(self):
        super(TrajSimi, self).__init__()

        self.dic_datasets = TrajSimi.load_trajsimi_dataset()
        self.space = create_cellspace(*self.dic_datasets['trajs_space_range'], 
                                      Config.cell_size, Config.cell_size, 
                                      Config.cellspace_buffer)
        logging.info('Cellspace: ' + str(self.space))
        
        # create dataloader
        len_dataset_trains = len(self.dic_datasets['trains_traj'])
        training_batch_size = min(len_dataset_trains, Config.trajsimi_batch_size)
        train_dataset = TrajSimiDatasetTraining(self.dic_datasets['trains_traj'], training_batch_size)
        
        duplicate_short_tolerance = Config.traj_duplicate_short_tolerance if Config.dataset == 'xian' else 0
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size = 1, 
                                           shuffle = False, 
                                           num_workers = 4, 
                                           collate_fn = partial(collate_training, 
                                                                space = copy.deepcopy(self.space),
                                                                duplicate_short_tolerance = duplicate_short_tolerance) )
        
        # ground-truth simi for training dataset
        self.dataset_simi_trains = self.dic_datasets['trains_simi']
        self.dataset_simi_trains = torch.tensor(self.dataset_simi_trains, device = Config.device, dtype = torch.float)
        self.dataset_simi_trains = (self.dataset_simi_trains + self.dataset_simi_trains.T) / self.dic_datasets['max_distance']
        
        self.checkpoint_filepath = '{}/{}_trajsimi_{}_{}_best{}.pt'.format(
                                            Config.checkpoint_dir,
                                            Config.dataset_prefix,
                                            Config.trajsimi_encoder_name,
                                            Config.trajsimi_measure_fn_name,
                                            Config.dumpfile_uniqueid)
        
        self.encoder = TSMini().to(Config.device)
            

    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("train_trajsimi start.@={:.3f}".format(training_starttime))

        self.criterion = nn.MSELoss().to(Config.device)

        learnable_params = [{'params': self.encoder.parameters(),
                            'lr': Config.trajsimi_learning_rate,
                            'weight_decay': Config.trajsimi_learning_weight_decay}]
            
        optimizer = torch.optim.Adam( learnable_params )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size = Config.trajsimi_training_lr_degrade_step, 
                                                    gamma = Config.trajsimi_training_lr_degrade_gamma)
        
        best_epoch = 0
        best_hr_eval = 0
        bad_counter = 0
        bad_patience = Config.trajsimi_training_bad_patience

        for i_ep in range(Config.trajsimi_epoch):
            _time_ep = time.time()
            train_losses = []
            train_gpus = []
            train_rams = []

            self.encoder.train()

            for i_batch, batch in enumerate( self.train_dataloader ):
                optimizer.zero_grad()

                trajs, trajs_len, sampled_idxs = batch
                sub_simi = self.dataset_simi_trains[sampled_idxs][:,sampled_idxs]
                
                embs = self.encoder(trajs, trajs_len)
                
                truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal = 1) == 1]
                pred_l1_simi = torch.cdist(embs, embs, 1)
                pred_rank = pred_l1_simi
                pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
                pred_l1_simi = pred_l1_simi * (truth_l1_simi.mean() / pred_l1_simi.mean())
                loss_wmse = torch.mean(torch.pow(pred_l1_simi-truth_l1_simi, 2) * (1-truth_l1_simi))
                
                n = pred_rank.shape[0]
                pred_rank_max = pred_rank.max(dim = -1)[0].unsqueeze(1)
                pred_rank = pred_rank.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
                pred_rank = pred_rank_max - pred_rank
                
                sub_simi_max = sub_simi.max(dim = -1)[0].unsqueeze(1)
                sub_simi = sub_simi.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
                sub_simi = sub_simi_max - sub_simi
                loss_rank = lambdaLoss(pred_rank, sub_simi, k = 10)

                loss = loss_wmse * Config.trajsimi_loss_mse_weight * 100 + \
                        loss_rank * (1 - Config.trajsimi_loss_mse_weight)
                    
                loss.backward()
                optimizer.step()
                
                train_losses.append([loss.item(), loss_wmse.item(), loss_rank.item()])
                train_gpus.append(tool_funcs.GPUInfo.mem()[0])
                train_rams.append(tool_funcs.RAMInfo.mem())

                if i_batch % 1000 == 0:
                    logging.info("training. ep-batch={}-{}, loss={:.4f}, @={:.3f}, gpu={}, ram={}".format( \
                                i_ep, i_batch, loss.item(), time.time()-_time_ep,
                                tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

            scheduler.step() 

            # i_ep
            loss_ep_avg = np.mean(np.asarray(train_losses), axis = 0).tolist()
            logging.info("training. i_ep={}, loss={:.4f}/{:.4f}/{:.4f}, @={:.3f}".format( \
                        i_ep, *loss_ep_avg, time.time()-_time_ep))
            
            _time_eval = time.time()
            eval_metrics = self.test(dataset_type = 'eval')
            logging.info("eval.     i_ep={}, mseloss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}, gpu={}, ram={}, @={:.3f}".format( \
                        i_ep, *eval_metrics, time.time()-_time_eval))
            
            hr_eval_ep = eval_metrics[4]
            training_gpu_usage = tool_funcs.mean(train_gpus)
            training_ram_usage = tool_funcs.mean(train_rams)

            # early stopping
            if  hr_eval_ep > best_hr_eval:
                best_epoch = i_ep
                best_hr_eval = hr_eval_ep
                bad_counter = 0
                torch.save({"encoder": self.encoder.state_dict()}, self.checkpoint_filepath)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == Config.trajsimi_epoch:
                training_endtime = time.time()
                logging.info("training end. @={:.3f}, best_epoch={}, best_hr_eval={:.4f}".format( \
                            training_endtime - training_starttime, best_epoch, best_hr_eval))
                break
            
        # test
        checkpoint = torch.load(self.checkpoint_filepath)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.encoder.eval()

        test_starttime = time.time()
        test_metrics = self.test(dataset_type = 'test')
        test_endtime = time.time()
        logging.info("test. @={:.3f}, mseloss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}, gpu={}, ram={}".format( \
                    test_endtime - test_starttime, *test_metrics))

        return {'task_train_time': training_endtime - training_starttime, \
                'task_train_gpu': training_gpu_usage, \
                'task_train_ram': training_ram_usage, \
                'task_test_time': test_endtime - test_starttime, \
                'task_test_gpu': test_metrics[7], \
                'task_test_ram': test_metrics[8], \
                'hr10':test_metrics[1], 'hr50':test_metrics[2], 'hr50in10':test_metrics[3], \
                'hr5':test_metrics[4], 'hr20':test_metrics[5], 'hr20in5':test_metrics[6]}


    @torch.no_grad()
    def test(self, dataset_type):
        # prepare dataset
        if dataset_type == 'eval':
            datasets_simi, max_distance = self.dic_datasets['evals_simi'], self.dic_datasets['max_distance']
            datasets = self.dic_datasets['evals_traj']
        elif dataset_type == 'test':
            datasets_simi, max_distance = self.dic_datasets['tests_simi'], self.dic_datasets['max_distance']
            datasets = self.dic_datasets['tests_traj']

        self.encoder.eval()

        datasets_simi = torch.tensor(datasets_simi, device = Config.device, dtype = torch.float)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance

        duplicate_short_tolerance = Config.traj_duplicate_short_tolerance if Config.dataset == 'xian' else 0
        dataloader = DataLoader(datasets,
                                batch_size = Config.trajsimi_batch_size, 
                                shuffle = False, 
                                num_workers = 4, 
                                collate_fn = partial(collate_eval_test, 
                                                     space = copy.deepcopy(self.space), 
                                                     duplicate_short_tolerance = duplicate_short_tolerance) )
        traj_outs = []
        for _, batch in enumerate(dataloader):
            trajs, trajs_len = batch
            embs = self.encoder(trajs, trajs_len)
            traj_outs.append(embs)
        
        # compute similarity
        traj_outs = torch.cat(traj_outs)
        pred_l1_simi = torch.cdist(traj_outs, traj_outs, 1)
        truth_l1_simi = datasets_simi
        pred_l1_simi_seq = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
        truth_l1_simi_seq = truth_l1_simi[torch.triu(torch.ones(truth_l1_simi.shape), diagonal = 1) == 1]

        # metrics
        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)
        hrA = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 10)
        hrB = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 50)
        hrBinA = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 10)
        hrC = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 5)
        hrD = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 20)
        hrDinC = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 5)
        gpu = tool_funcs.GPUInfo.mem()[0]
        ram = tool_funcs.RAMInfo.mem()

        return loss.item(), hrA, hrB, hrBinA, hrC, hrD, hrDinC, gpu, ram


    @staticmethod
    def hitting_ratio(preds: torch.Tensor, truths: torch.Tensor, pred_topk: int, truth_topk: int):
        # hitting ratio and recall metrics. see NeuTraj paper
        # the overlap percentage of the topk predicted results and the topk ground truth
        # overlap(overlap(preds@pred_topk, truths@truth_topk), truths@truth_topk) / truth_topk

        # preds = [batch_size, class_num], tensor, element indicates the probability
        # truths = [batch_size, class_num], tensor, element indicates the probability
        assert preds.shape == truths.shape and pred_topk < preds.shape[1] and truth_topk < preds.shape[1]

        _, preds_k_idx = torch.topk(preds, pred_topk + 1, dim = 1, largest = False)
        _, truths_k_idx = torch.topk(truths, truth_topk + 1, dim = 1, largest = False)

        preds_k_idx = preds_k_idx.cpu()
        truths_k_idx = truths_k_idx.cpu()

        tp = sum([np.intersect1d(preds_k_idx[i], truths_k_idx[i]).size for i in range(preds_k_idx.shape[0])])
        
        return (tp - preds.shape[0]) / (truth_topk * preds.shape[0])


    @staticmethod
    def load_trajsimi_dataset():
        # read (1) traj dataset for trajsimi, (2) simi matrix dataset for trajsimi
        trajsimi_traj_dataset_file = Config.dataset_file
        trajsimi_simi_dataset_file = '{}_traj_simi_dict_{}.pkl'.format( \
                                    Config.dataset_file, Config.trajsimi_measure_fn_name)

        trains_traj, evals_traj, tests_traj, trajs_merc_range = \
                read_trajsimi_traj_dataset(trajsimi_traj_dataset_file)
                
        trains_simi, evals_simi, tests_simi, max_distance = \
                read_trajsimi_simi_dataset(trajsimi_simi_dataset_file)
        
        # trains_traj : [[[lon, lat_in_merc], [], ..], [], ...]
        # trains_simi : list of list
        return {'trains_traj': trains_traj, 'evals_traj': evals_traj, 'tests_traj': tests_traj, \
                'trains_simi': trains_simi, 'evals_simi': evals_simi, 'tests_simi': tests_simi, \
                'max_distance': max_distance, 'trajs_space_range': trajs_merc_range}


def collate_training(batch, space, duplicate_short_tolerance):
    trajs, sampled_idxs = batch[0]

    trajs = [preprocess_traj(t, space, duplicate_short_tolerance) for t in trajs]
    trajs, trajs_len = padding_traj(trajs)
    
    trajs = torch.tensor(trajs, dtype = torch.float).to(Config.device)
    trajs_len = torch.tensor(trajs_len, dtype = torch.long, device = Config.device)
    
    return trajs, trajs_len, sampled_idxs


def collate_eval_test(trajs_src, space, duplicate_short_tolerance):
    traj_tensor_type = None
    trajs = [preprocess_traj(t, space, duplicate_short_tolerance) for t in trajs_src]
    traj_tensor_type = torch.float
    
    trajs, trajs_len = padding_traj(trajs)
    
    trajs = torch.tensor(trajs, dtype = traj_tensor_type).to(Config.device)
    trajs_len = torch.tensor(trajs_len, dtype = torch.long, device = Config.device)

    return  trajs, trajs_len

