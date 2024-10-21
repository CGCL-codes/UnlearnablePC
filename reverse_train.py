import os,sys,argparse,time,torch
from model import PointNet, DGCNN
import sklearn.metrics as metrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import importlib
import torch.optim as optim
from tqdm import tqdm 
from torch.utils.data import DataLoader, TensorDataset 
from utils.logging import Logging_str
from utils import set_seed,jitter_point_cloud,scale_point_cloud,class_wise_transformation,class_wise_reverse_transformation,get_list, SRSDefense, SORDefense
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import show_time, transform_time
import math, random
from data_utils.ModelNetDataLoader40 import ModelNetDataLoader40, pc_normalize
from data_utils.ModelNetDataLoader10 import ModelNetDataLoader10
from data_utils.ShapeNetDataLoader import PartNormalDataset
from data_utils.KITTIDataLoader import KITTIDataLoader
from data_utils.ScanObjectNNDataLoader import ScanObjectNNDataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'classifiers'))

 


def load_clean_train_data(args, data_path):
    if args.dataset == 'ModelNet40':
        DATASET = ModelNetDataLoader40(
            root=data_path,
            npoint=args.input_point_nums,
            split='train',
            normal_channel=False
        )
    elif args.dataset == 'ModelNet10':
        DATASET = ModelNetDataLoader10(
            root=data_path,
            npoint=args.input_point_nums,
            split='train',
            normal_channel=False
        )
    elif args.dataset == 'ShapeNetPart':
        DATASET = PartNormalDataset(
            root=data_path,
            npoint=args.input_point_nums,
            split='train',
            normal_channel=False
        )
    elif args.dataset == 'kitti':
        DATASET = KITTIDataLoader(
            root=data_path,
            npoints=256,
            split='train',
        )
    elif args.dataset == 'ScanObjectNN':
        DATASET = ScanObjectNNDataLoader(
            root=data_path,
            npoint=args.input_point_nums,
            split='train',
        )
    else:
        raise NotImplementedError

    T_DataLoader = torch.utils.data.DataLoader(
        DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,drop_last=True
    )
    return T_DataLoader



def data_preprocess(data):
    """Preprocess the given data and label.
    """
    points, target = data
    points = points # [B, N, C]
    target = target[:, 0] # [B]
    points = points.cuda()
    target = target.cuda().long()
    return points, target

def build_models(args):
    MODEL = importlib.import_module(args.target_model)
    classifier = MODEL.get_model(
        args.NUM_CLASSES,
        normal_channel=False
    )
    classifier = classifier.to(args.device)
    return classifier


def cal_loss(pred, gold, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    # gold = gold.view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')
    return loss


def main():
    if args.dataset == 'ModelNet40':
        args.NUM_CLASSES = 40
        args.data_path = "clean_data/modelnet40_normal_resampled"
        test_dataset = ModelNetDataLoader40(root=args.data_path, npoint=args.input_point_nums, split='test', normal_channel=False)
    elif args.dataset == 'ModelNet10':
        args.NUM_CLASSES = 10
        args.data_path = "clean_data/modelnet40_normal_resampled"
        test_dataset = ModelNetDataLoader10(root=args.data_path, npoint=args.input_point_nums, split='test', normal_channel=False)
    elif args.dataset == 'ShapeNetPart':
        args.NUM_CLASSES = 16
        args.data_path = "clean_data/shapenetcore_partanno_segmentation_benchmark_v0_normal"
        test_dataset = PartNormalDataset(root=args.data_path, npoint=args.input_point_nums, split='test', normal_channel=False)
    elif args.dataset == 'kitti':
        args.NUM_CLASSES = 2
        args.data_path = 'clean_data/KITTI/training/object_cloud'
        test_dataset = KITTIDataLoader(root=args.data_path, npoints=256, split='test')
    elif args.dataset == 'ScanObjectNN':
        args.NUM_CLASSES = 15
        args.data_path ='clean_data/h5_files'
        test_dataset = ScanObjectNNDataLoader(root=args.data_path, npoint=args.input_point_nums, split='test')

    print("Target model is {}".format(args.target_model))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,drop_last=True
    )


    if args.target_model == 'pointnet_cls':
        model = PointNet(args, output_channels=args.NUM_CLASSES).cuda()
    elif args.target_model == 'dgcnn':
        model = DGCNN(args, output_channels=args.NUM_CLASSES).cuda()
    else:
        model = build_models(args).cuda()
 
    train_loader = load_clean_train_data(args, args.data_path)
    
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=0.1*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    
    scheduler = CosineAnnealingLR(opt, args.epoch, eta_min=args.lr)
    criterion = cal_loss
    test_acc_list, train_acc_list = [], []
    

    import ast
    args.mode = ast.literal_eval(args.mode)
    mode_list = {m : get_list(m, args) for m in args.mode}


    for epoch in range(args.epoch):
        scheduler.step()
        train_loss, count = 0.0, 0.0
        model.train()
        train_pred, train_true = [], []

        for data in tqdm(train_loader):
            if args.dataset == 'ShapeNetPart':
                data = data[:2]
            data, label = data_preprocess(data)


            for idx in range(len(data)): 
                trans_data = data[idx].clone().detach()
                for k, v in mode_list.items(): 
                    trans_data = torch.tensor(class_wise_transformation(trans_data, k, v, int(label[idx].item())))   
                    trans_data = torch.tensor(class_wise_reverse_transformation(trans_data, k, v, int(label[idx].item())))


            data, label = data.cuda(), label.long().cuda().squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        round_acc = round(train_acc*100, 2)
        train_acc_list.append(round_acc)
        print('Epoch[%d] loss: %.4f, train acc: %.4f' % (epoch + 1, train_loss * 1.0 / count, train_acc))

        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []

        for data in tqdm(test_loader):
            if args.dataset == 'ShapeNetPart':
                data = data[:2] 
            data, label = data_preprocess(data)

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred) 
        round_acc = round(test_acc*100, 2)
        test_acc_list.append(round_acc)
        if (epoch + 1) % 10 == 0:
            print('\nEpoch[%d] loss: %.4f, test acc: %.2f\n' % (epoch + 1, test_loss * 1.0 / count, round_acc)) 

  
    import csv
    with open(os.path.join(f'rebuttal_results.csv'), 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["reverse", args.slight_range, args.main_range, args.sca_min, args.sca_max, args.dataset, args.target_model, round_acc])   
  

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unlearnable 3D Point Clouds')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--input_point_nums', type=int, default=1024, help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2023, metavar='S', help='random seed (default: 2022)')
    parser.add_argument('--dataset', type=str, default='ModelNet10',  choices=['ModelNet10', 'ModelNet40', 'ShapeNetPart', 'kitti', 'ScanObjectNN'])
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_workers', type=int, default=4,help='Worker nums of data loading.')
    parser.add_argument('--target_model', type=str, default='pointnet_cls',choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct', 'curvenet', 'simple_view', 'riconv2', 'gcn3d', 'rscnn','pointtransformerv3', 'pointmlp'])
    parser.add_argument('--defense_method', type=str, default=None,choices=['sor', 'srs', 'dupnet', 'lpf'])
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')
    parser.add_argument('--epoch', default=80, type=int, help='') 
    parser.add_argument('--use_sgd', action='store_true', help='sgd')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')  
    parser.add_argument('--slight_range', type=int, default=15, help='x,y angle range [para 1]')
    parser.add_argument('--main_range', type=int, default=120, help='z angle range [para 2]')
    parser.add_argument('--sca_min', type=float, default=0.6, help='scale min bound [para 3]')
    parser.add_argument('--sca_max', type=float, default=0.8, help='scale max bound [para 4]')
    parser.add_argument('--mode', type=str) 

    args = parser.parse_args()
    set_seed(args.seed)
    args.device = torch.device("cuda")

 
    main()