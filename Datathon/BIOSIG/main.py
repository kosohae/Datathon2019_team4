import os
import utils
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# dataset
from data import SignalDataset

# call model
from model import Baseline
import train
# base config


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default='./data', help='signal data dir')
    parser.add_argument('--num_epoch', type=int, default=10, help='epoch')
    parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='data batch size')
    parser.add_argument('--model', type=str,
                        default='Baseline', help='train model')
    parser.add_argument('--save_dir', type=str,
                        default='./results', help='output data dir')
    parser.add_argument('--save_name', type=str,
                        default='', help='manual name')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='Multi gpu training ')
    parser.add_argument('--device', type=None, default=torch.device('cuda:1'),
                        help='cuda device index')
    parser.add_argument('--mode', type=str, choices=['3000', '1000'], help='data numbers')
    parser.add_argument('--num_worker', type=int, default=4, help='num workers')

    args = parser.parse_args()
    return args


def split_weight(net):
    """
    split weights into categories
    one : conv, linear layer => decay
    others : bn weights, bias 
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)
        else:
            # class에 인자가 있는지 확인
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)
    # net.parameters() 형태로 반환
    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


def main():
    args = config()
    # args, inner variable
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_worker
    torch.backends.cudnn.benchmark = True

    # data loader - already SignalDataset to cuda
    # dataset : dictionary train, dev, test
    datasets = {}
    dataloaders = {}

    for k in ['train', 'eval', 'test']:
        datasets[k] = SignalDataset(k, args.data_dir)
        dataloaders[k] = DataLoader(
            datasets[k], args.batch_size, shuffle=True, num_workers=4)
        if k == 'test':
            dataloaders[k] = DataLoader(
                datasets[k], args.batch_size, shuffle=False, num_workers=4)

    # model load

    if args.ngpu > 1:
        print(f"Model Build....{args.model}")
        model = args.model().to(device)
        torch.nn.DataParallel(model)
    else:
        print(f"Model Build....{args.model}")
        model = Baseline().to(device)


    # criterion

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()

    # optimizer
    # adam default => le =1e-3 , betas : 0.9, 0.999 eps=1e-8, weight decay=0
    params = split_weight(model)
    #optimizer = optim.Adam(params)
    optimizer = optim.Adamax(params, lr=args.lr)
    # scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train
    best_model = train.train(dataloaders, model, criterion, optimizer, scheduler, args)

    # Test
    #test_loss, test_pred = test(dataloaders, model, criterion, optimizer, scheduler, args)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # save


if __name__ == '__main__':
    main()
