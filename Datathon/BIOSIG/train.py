import os
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
from main import config
from tqdm.auto import tqdm
#  base config
args = config()


# normally train & evaluation consistantly going together


def train(dataloaders, model, criterion, optimizer, scheduler, args):
    since = time.time()
    epochs = args.num_epoch
    save_dir = args.save_dir
    device = args.device

    # best model find
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-'*20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'eval']:
            if phase == 'tarin':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for i, (vt, gt, random) in tqdm(dataloaders[phase]):
                optimizer.zero_grad()
                model = model.to(device)

                if args.mode == '3000':
                    vt = vt.to(device=device, dtype=torch.float)
                    gt = gt.to(device=device, dtype=torch.float)
                    random = random.to(device=device, dtype=torch.float)
                else:
                    vt = torch.cat((vt[:, :3, :].narrow(2, 2000, 1000),vt[:, 3:, :].narrow(2, 0, 1000)), 1)
                    random = torch.cat((random[:, :3, :].narrow(2, 2000, 1000), random[:, 3:, :].narrow(2, 0, 1000)), 1)
                    vt = vt.to(device=device, dtype=torch.float)
                    gt = gt.to(device=device, dtype=torch.float)
                    random = random.to(device=device, dtype=torch.float)

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(vt)
                    loss = criterion(preds, gt)

                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() #* vt.size(0)
                running_correct += torch.sum(preds == gt.data)
            # epoch
        running_loss /= len(dataloaders)
        model.eval()
        eval_loss, _ = test(model, dataloaders['test'])
        model.train()

        print('{} {} Loss : {:.4f}'.format(phase, epoch, running_loss))

        # if phase == 'test' and epoch_acc > best_acc:
        #     best_acc = epoch_acc
        #     best_model = copy.deepcopy(model.state_dict())

    present = time.time() - since
    print('Time complete in {:.0f}m {:.0f}s'.format(present // 60, present % 60))
    # load best weight
    model.load_state_dict(best_model)
    return model


def test(model, dataloader, save_dir, args):

    device = args.device
    model_path = os.path.join(save_dir, 'model.pth')
    weight = torch.load(model_path)
    model.load_state_dict(weight)
    print(f'{model_path} loading the best model')

    cnt = 0
    loss = 0
    # roc, pearsonr
    preds = []
    gts = []
    for vt, gt, random in tqdm(dataloader):
        gts.extend(gt.data.numpy)
        if args.mode == '3000':
            vt = vt.to(device=device, dtype=torch.float)
            gt = gt.to(device=device, dtype=torch.float)
            random = random.to(device=device, dtype=torch.float)
        else:
            vt = torch.cat((vt[:, :3, :].narrow(2, 2000, 1000),vt[:, 3:, :].narrow(2, 0, 1000)), 1)
            random = torch.cat((random[:, :3, :].narrow(2, 2000, 1000), random[:, 3:, :].narrow(2, 0, 1000)), 1)
            vt = vt.to(device=device, dtype=torch.float)
            gt = gt.to(device=device, dtype=torch.float)
            random = random.to(device=device, dtype=torch.float)
        pred = model(vt)
        preds.extend(pred.data.detach().cpu().numpy())

        loss += F.binary_cross_entropy_with_logits(pred, gt)
        cnt += 1

    print('Pearsonr:\t', utils.get_pearsonr(gts, preds))
    print(f'R-score:{utils.get_roc_curve(gts, preds)}')

    return loss / cnt, preds