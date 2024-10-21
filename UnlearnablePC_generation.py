import argparse,random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from data_utils.ModelNetDataLoader40 import ModelNetDataLoader40
from data_utils.ModelNetDataLoader10 import ModelNetDataLoader10
from data_utils.ShapeNetDataLoader import PartNormalDataset
from data_utils.KITTIDataLoader import KITTIDataLoader
from data_utils.ScanObjectNNDataLoader import ScanObjectNNDataLoader
from torch.utils.data import DataLoader, TensorDataset
 
from utils import set_seed 
import math
from utils import class_wise_transformation,get_list
import os
import sys
import importlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'classifiers'))




def load_data(args, data_path):
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
    elif args.dataset == 'KITTI':
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
        num_workers=args.num_workers
    )
    print('Finish Loading Dataset...')
    return T_DataLoader

def data_preprocess(data):
    points, target = data

    points = points # [B, N, C]
    target = target[:, 0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target

def save_tensor_as_txt(args, points, filename):  
    points = points.squeeze(0)
    file_path = os.path.join(args.output_dir,'example')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(os.path.join(file_path,filename), "w") as f:
        for i in range(points.shape[0]):
            msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2])
            f.write(msg+'\n')
        f.close()
 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unleanrable 3D Point Clouds: Class-wise Transformation Is All You Need')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--input_point_nums', type=int, default=1024, help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2022, metavar='S', help='random seed (default: 2022)')
    parser.add_argument('--dataset', type=str, default='ModelNet10', choices=['ModelNet10', 'ModelNet40', 'ShapeNetPart', 'KITTI', 'ScanObjectNN'])
    parser.add_argument('--target_model', type=str, default='pointnet_cls', choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct', 'curvenet', 'simple_view'])
    parser.add_argument('--num_workers', type=int, default=4, help='Worker nums of data loading.')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--slight_range', type=int, default=15, help='x,y angle range [para 1]')
    parser.add_argument('--main_range', type=int, default=120, help='z angle range [para 2]')
    parser.add_argument('--sca_min', type=float, default=0.6, help='scale min bound [para 3]')
    parser.add_argument('--sca_max', type=float, default=0.8, help='scale max bound [para 4]')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--NUM_CLASSES', type=int, default=10)
 

    args = parser.parse_args()
    args.device = torch.device("cuda")

    set_seed(args.seed)
    if args.dataset == 'ModelNet40':
        args.NUM_CLASSES = 40
        data_path = "./clean_data/modelnet40_normal_resampled"
    elif args.dataset == 'ModelNet10':
        args.NUM_CLASSES = 10
        data_path = "./clean_data/modelnet40_normal_resampled"
    elif args.dataset == 'ShapeNetPart':
        args.NUM_CLASSES = 16
        data_path = './clean_data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    elif args.dataset == 'kitti':
        args.NUM_CLASSES = 2
        data_path = './clean_data/KITTI/training/object_cloud'
    elif args.dataset == 'ScanObjectNN':
        args.NUM_CLASSES = 15
        data_path ='./clean_data/h5_files'
    assert args.NUM_CLASSES != 0

    import ast
    args.mode = ast.literal_eval(args.mode)
    mode_list = {m : get_list(m, args) for m in args.mode}
    UMT_k = len(mode_list)

    if UMT_k == 2:
        data_path = os.path.join("./UE", args.dataset, str(args.slight_range) + '_' + str(args.main_range) + '_' + str(args.sca_min) + '_' + str(args.sca_max), "example")
    elif UMT_k == 3:
        data_path = os.path.join("./UE", args.dataset, str(args.slight_range) + '_' + str(args.main_range) + '_' + str(args.sca_min) + '_' + str(args.sca_max) + '_' + str(20), "example")
    elif UMT_k == 4:
        data_path = os.path.join("./UE", args.dataset, str(args.slight_range) + '_' + str(args.main_range) + '_' + str(args.sca_min) + '_' + str(args.sca_max) + '_' + str(20) + '_' + str(0.4), "example")
    elif UMT_k == 1:
        data_path = os.path.join("./UE", args.dataset, str(args.slight_range) + '_' + str(args.main_range) + "_1.0_1.0", "example")

    data_loader = load_data(args, data_path) 
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))


    for batch_id, data in pbar:
        if args.dataset == 'ShapeNetPart':
            data = data[:2]
        data, label = data_preprocess(data) 
        for idx in range(len(data)):
            trans_data = data[idx].clone().detach()
            for k, v in mode_list.items():
                trans_data = torch.tensor(class_wise_transformation(trans_data, k, v, label[idx].item()))
            data[idx] = trans_data 
        save_tensor_as_txt(args, data.detach().cpu().numpy(), f'{batch_id}_transform_{label.item()}.txt')
 
    print("Generated Successfully!")