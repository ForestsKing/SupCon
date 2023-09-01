import argparse

import numpy as np
import torch

from exp.classify import ExpClassify
from utils.seed import setSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--loss', type=str, default='SupCon', help='[CrossEntropy, SimCLR, SupCon]')

    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    parser.add_argument('--download', type=bool, default=False)

    parser.add_argument('--itr', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--devices', type=int, default=0)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('\n===================== Args ========================')
    print(args)
    print('===================================================\n')

    setSeed(args.random_seed)

    acc = []
    for ii in range(args.itr):
        setting = "{0}_{1}".format(args.loss, ii)

        exp_classify = ExpClassify(args, setting)
        exp_classify.train()
        acc.append(exp_classify.test())

        torch.cuda.empty_cache()

    print('Accuracy: {0:.4f} ± {1:.4f}'.format(np.mean(acc), np.std(acc)))
    print('Done!')
