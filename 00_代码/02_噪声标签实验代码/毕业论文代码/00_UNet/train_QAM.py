import os, sys
import numpy as np
import pickle, h5py, time, argparse, itertools, datetime

import torch
import torch.nn as nn
import torch.utils.data
from libs.loss import WeightedCELoss
from libs.dataset import CTDataset

from libs.se_model import seg_module
from libs.unet_model import UNet
from libs.qua_model import se_res_extracter as qua_module

from tensorboardX import SummaryWriter

np.set_printoptions(precision=4,suppress=True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_args():
    parser = argparse.ArgumentParser(description='Quality Awareness Model')
    parser.add_argument('-o','--output', default='result/',
                        help='Output path')
    parser.add_argument('-n','--noise',type=float, default=0.50,help='Noise percentage')
    
    parser.add_argument('-kernel',type=float, default=9,help='Noise kernel')
    parser.add_argument('-ker-iter',type=int, default=2,
                        help='kernel iterations')
    
    parser.add_argument('-nosiyType',type=int, default=0,
                        help='0 for dilation, 1 for erosion')
    
    
    parser.add_argument('-lr', type=float, default=.0001,
                        help='Learning rate')
    parser.add_argument('-mi','--max-iter', type=int, default=10002,
                        help='Total number of iteration')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=251,
                        help='# of the epochs')
    parser.add_argument('--n_classes', nargs='?', type=int, default=2,
                        help='# of the classes')
    parser.add_argument('--iter-save', type=int, default=100,
                        help='Number of iteration to save')
    parser.add_argument('--epoch-save', type=int, default=10,
                        help='Number of iteration to save')
    parser.add_argument('-g','--num-gpu', type=int,  default=2,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=4,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=16,
                        help='Batch size')
    args = parser.parse_args()
    return args

def init(args):
    
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device

def dice(pred, gt, label):
    intersection = ((gt == label) * (pred == label)).sum().astype(np.float)
    
    return 2 * intersection / ((gt == label).sum() + (pred == label).sum()).astype(np.float)

def evalue(pred, labels, n_classes):
    
    lbl0 = (labels == 0).astype(np.int)
    lbl1 = (labels == 1).astype(np.int)
    pred0 = (pred == 0).astype(np.int)
    pred1 = (pred == 1).astype(np.int)
    #print((pred == 1).astype(np.int).sum())
    TP = (lbl1 * pred1).sum().astype(np.float)
    TN = (lbl0 * pred0).sum().astype(np.float)
    FN = lbl1.sum() - TP
    FP = pred1.sum() - TP
    F1 = 2*TP/(2*TP+FP+FN)
    ACC = (TP+TN)/(TP+TN+FN+FP)
    MIOU = ((TP/(TP+FP+FN))+(TN/(TN+FN+FP)))/2
    
    return ACC, MIOU, F1


def eval_valid(loader, model, device):
    model.eval()
    dval = 0
    ACC = 0
    acc = 0
    MIOU = 0
    miou = 0
    F1 = 0
    f1 = 0
    Dice = 0
    dcnt = len(loader)
    index = 0
    for i, (data, label,img_name) in enumerate(loader):
        data = data.to(device)
        pred = model(data)
        pred = pred.data.cpu().numpy()
        label = label.data.numpy()
        pred = np.argmax(pred, axis=1)
        Dice = dice(pred, label, 1)
        ACC, MIOU, F1 = evalue(pred, label, 2)  
        acc += ACC
        
        if ACC != 1:    
            f1 += F1
            index = index + 1
            dval += Dice
            miou += MIOU
        
        print('accuracy:', ACC)
        print('MIOU:', MIOU)
        print('F1 score:', F1)
        print('Dice:', Dice)
    
    print('Over Accuracy:', (acc / dcnt))
    print('Over MIOU:', (miou / index))
    print('Over F1 score:', (f1 / index))
    print('Over Dice:', (dval / index))
    return (dval / index),(acc / dcnt),(miou / index),(f1 / index)

def load_data(dname):
    train_label = np.array(h5py.File(dname, 'r')['label'],dtype=np.float16)
    train_input = np.array(h5py.File(dname, 'r')['image'],dtype=np.float16)
    return train_input,train_label

def get_input(args):
    
    '''
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
    
    '''
    
    data_path = "/home/chuang.lin/Desktop/graduation/dataset/IAI/"
    
    train_dataset = CTDataset(data_path, mode='train', 
                    noise_level=args.noise, noise_range=args.kernel, nosiyType=args.nosiyType, ker_iter=args.ker_iter)
    valid_dataset = CTDataset(data_path,mode='val')
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


best_acc = 0
lastest_acc = 0
lastest_epoch = 0

def train(args, train_loader, valid_loader, m1, m2, device, criterion, optimizer, logger, writer, epoch):
    clean_weight = 0
    noisy_weight = 0
    iter_cnt = 0
    all_loss = 0
    global best_acc, lastest_acc, lastest_epoch
    m1, m2 = m1, m2
    m1.train()
    m2.train()
    o1, o2 = optimizer
    

    for i, (data, label, qua, nlevel, weight) in enumerate(train_loader):
        iter_cnt += 1
        data, label, qua = data.to(device), label.to(device), qua.to(device)
        class_weight = weight.to(device)
        outscore = m1(data)
        outweight = m2(qua)
        outweight = outweight/outweight.sum()
        loss = criterion(outscore, label.long()) 
        loss = torch.mean(torch.sum(loss/args.batch_size,dim=0))
        wei_ = np.array([nlevel.data.cpu().numpy().reshape([-1]),
                         outweight.cpu().detach().numpy().reshape([-1])*args.batch_size]).T 
        
        c_data = wei_[wei_[:,0] == 0]
        n_data = wei_[wei_[:,0] > 0]   
        
        print('[Epoch %d / Iter %d] loss: %.4f Ave weight for clean data %.4f noisy data %.4f' 
                %(epoch, iter_cnt, loss.item(),c_data[:,1].mean(),n_data[:,1].mean()))
        o1.zero_grad()
        o2.zero_grad()
        loss.backward()
        o1.step()
        o2.step()
        
        #logger.write("[Iteration %d] train_loss=%0.4f lr=%.5f" % (iter_cnt, \
        #           loss.item(), o1.param_groups[0]['lr']))
        logger.write("[Iteration %d] Ave weight for clean data %.4f noisy data %.4f" % (iter_cnt, \
                    c_data[:,1].mean(), n_data[:,1].mean()))
        
        clean_weight += c_data[:,1].mean()
        noisy_weight += n_data[:,1].mean()
        
        all_loss += loss.item()
    
    print('Ave weight for clean data %.4f noisy data %.4f' %(clean_weight/iter_cnt, noisy_weight/iter_cnt))
    writer.add_scalar('train_loss', all_loss/iter_cnt, epoch)
    
    writer.add_scalars('weight', {'clean_weight': clean_weight/iter_cnt,
                                 'nosiy_weight': noisy_weight/iter_cnt}, epoch)
        
    if not os.path.exists(args.output+'/saveModule'):
        os.makedirs(args.output+'/saveModule')
    dval,acc,miou,f1 = eval_valid(valid_loader, m1, device)
    logger.write("[Epoch %d] dval=%0.4f acc=%.5f miou=%.5f f1 score=%.5f" % (epoch, \
                    dval, acc,miou,f1))
    writer.add_scalar('F1_Score', f1, epoch)
    writer.add_scalar('Pixel Accuracy', acc, epoch)
    writer.add_scalar('MIOU', miou, epoch)
    m1.train()
    if epoch % args.epoch_save == 0:
        torch.save(m1.state_dict(), args.output+('/saveModule/%d_seg_module.pth'%epoch))
        
                 
    if acc > best_acc:
        best_acc = acc
        print('------save best---------')
        torch.save(m1.state_dict(), args.output+'/saveModule/{}_{}_{}.pth'.format('best', epoch, best_acc))
        
        if os.path.isfile(args.output+'/saveModule/{}_{}_{}.pth'.format('best', lastest_epoch, lastest_acc)):
            os.remove(args.output+'/saveModule/{}_{}_{}.pth'.format('best', lastest_epoch, lastest_acc))
        
        lastest_acc = best_acc
        lastest_epoch = epoch

def main():
    
    os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
    
    args = get_args()
    

    print('0. initial setup')
    device = init(args) 
    logger, writer = get_logger(args)

    print('1. setup data')
    train_loader, valid_loader = get_input(args)

    print('2. setup model')
    m1 = UNet(n_classes=2,n_channels=3)
    m2 = qua_module(in_num=4)
    
    if args.num_gpu>1: 
        m1 = nn.DataParallel(m1)
        m2 = nn.DataParallel(m2)
    if torch.cuda.is_available():
        m1.cuda()
        m2.cuda()
    criterion = WeightedCELoss()

    print('3. setup optimizer')
    
    optimizer = [torch.optim.Adam(m1.parameters(), lr=args.lr, betas=(0.9, 0.999)),
                 torch.optim.Adam(m2.parameters(), lr=args.lr, betas=(0.9, 0.999))]
    
    print('4. start training')
    for epoch in range(args.n_epoch):
        train(args, train_loader, valid_loader, m1, m2, device, criterion, optimizer, logger, writer, epoch)
    
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()
