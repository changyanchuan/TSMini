import sys
import logging
import argparse
import torch

from config import Config
from utils import tool_funcs
from task.trajsimi import TrajSimi

def parse_args():
    # dont set default values here! config.py is used to set default values.
    parser = argparse.ArgumentParser(description = "train_trajsimi.py")
    parser.add_argument('--dumpfile_uniqueid', type = str, help = 'see config.py')
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dataset', type = str, help = '')
    parser.add_argument('--trajsimi_measure_fn_name', type = str, help = '')
    
    parser.add_argument('--tsmin_trans_attention_layer', type = int, help = '')
    parser.add_argument('--trajsimi_loss_mse_weight', type = float, help = '')
    parser.add_argument('--trajsimi_batch_size', type = int, help = '')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def main():
    enc_name = Config.trajsimi_encoder_name
    fn_name = Config.trajsimi_measure_fn_name
    metrics = tool_funcs.Metrics()

    task = TrajSimi()
    metrics.add(task.train())

    logging.info('[EXPFlag]model={},dataset={},fn={},{}'.format( \
                enc_name, Config.dataset_prefix, fn_name, str(metrics)))
    return


# nohup python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name frechet &> result &
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    Config.update(parse_args())

    logging.basicConfig(level = logging.DEBUG if Config.debug else logging.INFO,
                        format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'), 
                                    logging.StreamHandler()]
                        )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    main()
    