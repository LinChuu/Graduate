import os, sys
import numpy as np
import pickle, h5py, time, argparse, itertools, datetime

import torch
import torch.nn as nn
import torch.utils.data
from libs.loss import WeightedCELoss
from libs.dataset import CTDataset

from libs.se_model import seg_module
from libs.qua_model import se_res_extracter as qua_module

from tensorboardX import SummaryWriter

np.set_printoptions(precision=4,suppress=True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_args():
    parser = argparse.ArgumentParser(description='Quality Awareness Model')
    parser.add_argument('-o','--output', default='result/',
                        help='Output path')
    parser.add_argument('-n','--noise',type=float, default=0.5,help='Noise percentage')
    parser.add_argument('-ns',type=float, default=5,help='Noise range (min)')
    parser.add_argument('-nm',type=float, default=13,help='Noise range (max)')
    parser.add_argument('-lr', type=float, default=.0001,
                        help='Learning rate')
    parser.add_argument('-mi','--max-iter', type=int, default=10002,
                        help='Total number of iteration')
    parser.add_argument('--iter-save', type=int, default=100,
                        help='Number of iteration to save')
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=4,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=32,
                        help='Batch size')
    args = parser.parse_args()
    return args

def init(args):
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def dice(gt, pred, label):
    intersection = ((gt == label) * (pred == label)).sum().astype(np.float)
    return 2 * intersection / ((gt == label).sum() + (pred == label).sum()).astype(np.float)

def eval_valid(loader, model, device):
    model.eval()
    dval = np.zeros(3)
    dcnt = len(loader)
    for i, (data, label) in enumerate(loader):
        data = data.to(device)
        pred = model(data)
        pred = pred.data.cpu().numpy()
        label = label.data.numpy()
        pred = np.argmax(pred, axis=1)
        label = label.reshape([-1])
        pred = pred.reshape([-1])
        for j in range(3):
            dval[j] += dice(pred, label, j+1)
    print('Class-wise dice: ', dval / dcnt)
    print('Overall dice:', np.mean(dval / dcnt))
    return np.mean(dval / dcnt)

def load_data(dname):
    train_label = np.array(h5py.File(dname, 'r')['label'],dtype=np.float16)
    train_input = np.array(h5py.File(dname, 'r')['image'],dtype=np.float16)
    return train_input,train_label

def get_input(args):
    with h5py.File('data/0_fold_0.h5','r') as fid:
        train_input = np.array(fid['image'])
        train_label = np.array([fid['lung'],fid['heart'],fid['clavicle']])
    with h5py.File('data/0_fold_1.h5','r') as fid:      
        train_input = np.concatenate([train_input,np.array(fid['image'])],axis=0)
        train_label = np.concatenate([train_label,
                            np.array([fid['lung'],fid['heart'],fid['clavicle']])],axis=1)    
    with h5py.File('data/0_fold_2.h5','r') as fid:      
        valid_input = np.array(fid['image'])
        valid_label = np.array([fid['lung'],fid['heart'],fid['clavicle']])
    train_label = np.transpose(train_label,[1,0,2,3])
    valid_label = np.transpose(valid_label,[1,0,2,3])
    train_dataset = CTDataset(input=train_input, label=train_label,mode='train', 
                    noise_level=args.noise, noise_range=[args.ns, args.nm])
    valid_dataset = CTDataset(input=valid_input, label=valid_label,mode='valid')
    train_loader =  torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_cpu, pin_memory=True)
    valid_loader =  torch.utils.data.DataLoader(valid_dataset,
            batch_size=1, shuffle=False,
            num_workers=args.num_cpu, pin_memory=True)
    return train_loader, valid_loader

def get_logger(args):
    log_name = args.output+'/log'
    date = str(datetime.datetime.now()).split(' ')[0]
    time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
    log_name += '_approx_'+date+'_'+time
    logger = open(log_name+'.txt','w') # unbuffered, write instantly

    # tensorboardX
    writer = SummaryWriter('runs/'+log_name)
    return logger, writer

def train(args, train_loader, valid_loader, model, device, criterion, optimizer, logger, writer):
    iter_cnt = 0
    m1, m2 = model
    m1.train()
    m2.train()
    o1, o2 = optimizer
    dvalue = 0

    for i, (data, label, qua, nlevel, weight) in enumerate(train_loader):
        iter_cnt += 1
        data, label, qua = data.to(device), label.to(device), qua.to(device)
        class_weight = weight.to(device)
        outscore = m1(data)
        outweight = m2(qua)
        loss = criterion(outscore, label.long(), class_weight)
        loss = torch.mean(torch.sum(loss * outweight,dim=0))
        wei_ = np.array([nlevel.data.cpu().numpy().reshape([-1]),
                         outweight.cpu().detach().numpy().reshape([-1])*args.batch_size]).T    
        c_data = wei_[wei_[:,0] == 0]
        n_data = wei_[wei_[:,0] > 0]
        print('[Iter %d] loss: %.4f Ave weight for clean data %.4f noisy data %.4f' 
                %(iter_cnt, loss.item(),c_data[:,1].mean(),n_data[:,1].mean()))
        o1.zero_grad()
        o2.zero_grad()
        loss.backward()
        o1.step()
        o2.step()

        logger.write("[Iteration %d] train_loss=%0.4f lr=%.5f" % (iter_cnt, \
                    loss.item(), o1.param_groups[0]['lr']))
        writer.add_scalar('train_loss', loss.item(), iter_cnt)
        
        if i % args.iter_save == 0:
            dval = eval_valid(valid_loader, m1, device)
            m1.train()
            torch.save(m1.state_dict(), args.output+('%d_seg_module.pth'%iter_cnt))
            if dval > dvalue:
                dvalue = dval

def main():
    args = get_args()
    assert args.ns < args.nm

    print('0. initial setup')
    device = init(args) 
    logger, writer = get_logger(args)

    print('1. setup data')
    train_loader, valid_loader = get_input(args)

    print('2. setup model')
    model = [seg_module(n_channels=1,n_classes=4,train=True),qua_module(in_num=4)]
            
    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu))
    model = [m.to(device) for m in model]
    criterion = WeightedCELoss()

    print('3. setup optimizer')
    optimizer = [torch.optim.Adam(model[0].parameters(), lr=args.lr, betas=(0.9, 0.999)),
                 torch.optim.Adam(model[1].parameters(), lr=args.lr, betas=(0.9, 0.999))]

    print('4. start training')
    train(args, train_loader, valid_loader, model, device, criterion, optimizer, logger, writer)
  
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()
