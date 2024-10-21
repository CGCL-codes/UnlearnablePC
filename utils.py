import os
import numpy as np
import torch
import random
import math 
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import datetime
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
 
 

def test(model, loader):
    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    return accuracy

def compute_cat_iou(pred,target,iou_tabel):
    iou_list = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]
        batch_target = target[j]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
        for cat in np.unique(batch_target):
            # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            # iou = intersection/union if not union ==0 else 1
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
            iou_list.append(iou)
    return iou_tabel,iou_list

def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):
        part_ious = []
        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

def test_partseg(model, loader, catdict, num_classes = 50,forpointnet2=False):
    ''' catdict = {0:Airplane, 1:Airplane, ...49:Table} '''
    iou_tabel = np.zeros((len(catdict),3))
    iou_list = []
    metrics = defaultdict(lambda:list())
    hist_acc = []
    # mean_correct = []
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point,_= points.size()
        points, label, target, norm_plt = Variable(points.float()),Variable(label.long()), Variable(target.long()),Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(), label.squeeze().cuda(), target.cuda(), norm_plt.cuda()
        if forpointnet2:
            seg_pred = model(points, norm_plt, to_categorical(label, 16))
        else:
            labels_pred, seg_pred, _  = model(points,to_categorical(label,16))
            # labels_pred_choice = labels_pred.data.max(1)[1]
            # labels_correct = labels_pred_choice.eq(label.long().data).cpu().sum()
            # mean_correct.append(labels_correct.item() / float(points.size()[0]))
        # print(pred.size())
        iou_tabel, iou = compute_cat_iou(seg_pred,target,iou_tabel)
        iou_list+=iou
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        seg_pred = seg_pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['inctance_avg_iou'] = np.mean(iou_list)
    # metrics['label_accuracy'] = np.mean(mean_correct)
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
    metrics['class_avg_iou'] = np.mean(cat_iou)

    return metrics, hist_acc, cat_iou

def test_semseg(model, loader, catdict, num_classes = 13, pointnet2=False):
    iou_tabel = np.zeros((len(catdict),3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()
        points, target = Variable(points.float()), Variable(target.long())
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        if pointnet2:
            pred = model(points[:, :3, :], points[:, 3:, :])
        else:
            pred, _ = model(points)
        # print(pred.size())
        iou_tabel, iou_list = compute_cat_iou(pred,target,iou_tabel)
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    # print(iou_tabel)
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, hist_acc, cat_iou


def compute_avg_curve(y, n_points_avg):
    avg_kernel = np.ones((n_points_avg,)) / n_points_avg
    rolling_mean = np.convolve(y, avg_kernel, mode='valid')
    return rolling_mean
 

 
 
def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    # jittered_data += batch_data
    jittered_data = np.add(jittered_data, batch_data)
    return jittered_data

def scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data
    
def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)

    x = np.random.uniform() * 2 * np.pi
    y = np.random.uniform() * 2 * np.pi
    z = np.random.uniform() * 2 * np.pi

    x_matrix = np.array([[1, 0, 0], [0, np.cos(x), np.sin(x)], [0, -np.sin(x), np.cos(x)]])
    y_matrix = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
    z_matrix = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z), np.cos(z), 0], [0, 0, 1]])
    rotation_matrix = np.dot(x_matrix, y_matrix)
    rotation_matrix = np.dot(rotation_matrix, z_matrix)


    for k in range(batch_data.shape[0]): 
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def get_list(mode, args):
    list = []
    if mode == 'rot':
        Avg_num = math.ceil(args.NUM_CLASSES ** (1 / 3))
        x_list, y_list = [random.uniform(0, args.slight_range) for _ in range(Avg_num)], [random.uniform(0, args.slight_range) for _ in range(Avg_num)]
        z_list = [random.uniform(0, args.main_range) for _ in range(Avg_num)]

        for i in range(Avg_num):
            for j in range(Avg_num):
                for k in range(Avg_num):
                    list.append([x_list[i], z_list[j], y_list[k]])

        list = random.sample(list, args.NUM_CLASSES)

    elif mode == 'scale':
        list = [random.uniform(args.sca_min, args.sca_max) for _ in range(args.NUM_CLASSES)]

    elif mode == 'shear':
        list = [random.uniform(0, 0.4) for _ in range(args.NUM_CLASSES)]

    elif mode == 'twist':
        list = [random.uniform(0, 20) for _ in range(args.NUM_CLASSES)]

    elif mode == 'taper':
        list = [random.uniform(0, 50) for _ in range(args.NUM_CLASSES)]

    elif mode == 'translation':
        list = [[random.uniform(0, 0.3), random.uniform(0, 0.3), random.uniform(0, 0.3)] for _ in range(args.NUM_CLASSES)]
    return list 
 
 


 

def class_wise_reverse_transformation(point_cloud, mode, list, label):
    if torch.is_tensor(point_cloud):
        point_cloud = point_cloud.clone().detach().cpu().numpy()
    if mode == 'rot':
        x = np.pi / 180. * list[label][0]
        y = np.pi / 180. * list[label][1]
        z = np.pi / 180. * list[label][2]

        x_matrix = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        y_matrix = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        z_matrix = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        
        matrix = np.dot(np.dot(x_matrix, y_matrix), z_matrix)   
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc

    elif mode == 'scale':
        scaling_factor = list[label]
        reverse_sf = 1/list[label]
        matrix = np.array([ [reverse_sf, 0, 0], [0, reverse_sf, 0], [0, 0, reverse_sf] ])
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc

    elif mode == 'shear':
        a = list[label][0]
        b = list[label][1]
        matrix = np.array([[1, 0, 0],[0, 1, 0],[-a, -b, 1]])
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc   
 
    elif mode == 'twist':
        angle = np.pi / 180 * list[label]
        costz =  np.cos(point_cloud[:, 2] * angle)
        sintz = np.sin(point_cloud[:,2] * angle)    
        output = np.zeros_like(point_cloud)
        output[:,0] = point_cloud[:, 0] * costz + point_cloud[:, 1] * sintz 
        output[:,1] = point_cloud[:, 0] * (-sintz) + point_cloud[:, 1] * costz
        output[:,2] = point_cloud[:,2]
        return output

    else:
        raise NotImplementedError



def class_wise_transformation(point_cloud, mode, list, label):
    if torch.is_tensor(point_cloud):
        point_cloud = point_cloud.clone().detach().cpu().numpy()
    if mode == 'rot':
        x = np.pi / 180. * list[label][0]
        y = np.pi / 180. * list[label][1]
        z = np.pi / 180. * list[label][2]

        x_matrix = np.array([[1, 0, 0], [0, np.cos(x), np.sin(x)], [0, -np.sin(x), np.cos(x)]])
        y_matrix = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
        z_matrix = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z), np.cos(z), 0], [0, 0, 1]])
        
        matrix = np.dot(np.dot(x_matrix, y_matrix), z_matrix)   
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc

    elif mode == 'scale':
        matrix = np.array([[list[label], 0, 0], [0, list[label], 0], [0, 0, list[label]]])
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc

    elif mode == 'shear':
        a = list[label][0]
        b = list[label][1]
        matrix = np.array([[1, 0, 0],[0, 1, 0],[a, b, 1]])
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc   
 
    elif mode == 'twist':
        angle = np.pi / 180 * list[label]
        costz =  np.cos(point_cloud[:, 2] * angle)
        sintz = np.sin(point_cloud[:,2] * angle)    
        output = np.zeros_like(point_cloud)
        output[:,0] = point_cloud[:, 0] * costz - point_cloud[:, 1] * sintz 
        output[:,1] = point_cloud[:, 0] * sintz + point_cloud[:, 1] * costz
        output[:,2] = point_cloud[:,2]
        return output

    elif mode == 'taper':
        factor = np.pi / 180. * list[label] * point_cloud[:, 2] + 1
        matrix = np.array([[factor, 0, 0],
                            [0, factor, 0],
                            [0, 0, 1]])
        output = np.zeros_like(point_cloud)
        output[:,0] = factor * point_cloud[:, 0]
        output[:,1] = factor * point_cloud[:, 1]
        output[:,2] = point_cloud[:,2]
        return output
        
    elif mode == 'translation':
        point_cloud[:, 0] += list[label][0]
        point_cloud[:, 1] += list[label][1]
        point_cloud[:, 2] += list[label][2]
        return point_cloud

    else:
        raise NotImplementedError


def universe_transformation(point_cloud, mode):
    if torch.is_tensor(point_cloud):
        point_cloud = point_cloud.clone().detach().cpu().numpy()
    if mode == 'rot':
        x = np.pi / 180. * random.uniform(0, 20) 
        y = np.pi / 180. * random.uniform(0, 20) 
        z = np.pi / 180. * random.uniform(0, 20) 

        x_matrix = np.array([[1, 0, 0], [0, np.cos(x), np.sin(x)], [0, -np.sin(x), np.cos(x)]])
        y_matrix = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
        z_matrix = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z), np.cos(z), 0], [0, 0, 1]])
        rotation_matrix = np.dot(x_matrix, y_matrix)
        rotation_matrix = np.dot(rotation_matrix, z_matrix)   
        transformed_pc = np.matmul(point_cloud, rotation_matrix).astype('float32')
        return transformed_pc

    elif mode == 'shear':
        a = 0.4
        b = 0.4
        matrix = np.array([[1, 0, 0],[0, 1, 0],[a, b, 1]])
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc

    elif mode == 'scale':
        factor = random.uniform(0.6, 0.8)
        matrix = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, factor]])
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc

    elif mode == 'twist':
        angle = np.pi / 180 * 50
        costz =  np.cos(point_cloud[:, 2] * angle)
        sintz = np.sin(point_cloud[:,2] * angle)
        output = np.zeros_like(point_cloud)
        output[:,0] = point_cloud[:, 0] * costz - point_cloud[:, 1] * sintz 
        output[:,1] = point_cloud[:, 0] * sintz + point_cloud[:, 1] * costz
        output[:,2] = point_cloud[:,2]

        return output

    elif mode == 'taper':
        factor = np.pi / 180. * 50  * point_cloud[:, 2] + 1
        output = np.zeros_like(point_cloud)
        output[:,0] = factor * point_cloud[:, 0]
        output[:,1] = factor * point_cloud[:, 1]
        output[:,2] = point_cloud[:,2]
        return output

    elif mode == 'translation':
        dx = random.uniform(0, 0.3) 
        dy = random.uniform(0, 0.3) 
        dz = random.uniform(0, 0.3) 
        point_cloud[:, 0] += dx
        point_cloud[:, 1] += dy
        point_cloud[:, 2] += dz
        return point_cloud

    elif mode == 'reflection':
        matrix = np.array([[1, 0, 0],[0, -1, 0],[0, 0, 1]])
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc

    else:
        raise NotImplementedError

 

def random_transformation(point_cloud, mode):
    point_cloud = point_cloud.clone().detach().cpu().numpy()
    if mode == 'rot':
        x = np.pi / 180. * random.uniform(0, 360) 
        y = np.pi / 180. * random.uniform(0, 360) 
        z = np.pi / 180. * random.uniform(0, 360) 

        x_matrix = np.array([[1, 0, 0], [0, np.cos(x), np.sin(x)], [0, -np.sin(x), np.cos(x)]])
        y_matrix = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
        z_matrix = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z), np.cos(z), 0], [0, 0, 1]])
        rotation_matrix = np.dot(x_matrix, y_matrix)
        rotation_matrix = np.dot(rotation_matrix, z_matrix)   
        transformed_pc = np.matmul(point_cloud, rotation_matrix).astype('float32')
        return transformed_pc

    elif mode == 'shear':
        a = random.uniform(0, 0.4) 
        b = random.uniform(0, 0.4) 
        matrix = np.array([[1, 0, 0],[0, 1, 0],[a, b, 1]])
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc

    elif mode == 'scale':
        factor = random.uniform(0.8, 1.25)
        matrix = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, factor]])
        transformed_pc = np.matmul(point_cloud, matrix).astype('float32')
        return transformed_pc

    elif mode == 'twist':
        angle = np.pi / 180 * random.uniform(0, 50)
        costz =  np.cos(point_cloud[:, 2] * angle)
        sintz = np.sin(point_cloud[:,2] * angle)
        output = np.zeros_like(point_cloud)
        output[:,0] = point_cloud[:, 0] * costz - point_cloud[:, 1] * sintz 
        output[:,1] = point_cloud[:, 0] * sintz + point_cloud[:, 1] * costz
        output[:,2] = point_cloud[:,2]

        return output

    elif mode == 'taper':
        factor = np.pi / 180. * random.uniform(0,50)  * point_cloud[:, 2] + 1
        output = np.zeros_like(point_cloud)
        output[:,0] = factor * point_cloud[:, 0]
        output[:,1] = factor * point_cloud[:, 1]
        output[:,2] = point_cloud[:,2]
        return output

    elif mode == 'translation':
        dx = random.uniform(0, 0.3) 
        dy = random.uniform(0, 0.3) 
        dz = random.uniform(0, 0.3) 
        point_cloud[:, 0] += dx
        point_cloud[:, 1] += dy
        point_cloud[:, 2] += dz
        return point_cloud


    else:
        print(mode)
        raise NotImplementedError




def set_seed(seed):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")



class SORDefense(nn.Module):
    """Statistical outlier removal as defense.
    """

    def __init__(self, k=2, alpha=1.1, npoint=1024):
        """SOR defense.

        Args:
            k (int, optional): kNN. Defaults to 2.
            alpha (float, optional): \miu + \alpha * std. Defaults to 1.1.
        """
        super(SORDefense, self).__init__()

        self.k = k
        self.alpha = alpha
        self.npoint = npoint

    def outlier_removal(self, x):
        """Removes large kNN distance points.

        Args:
            x (torch.FloatTensor): batch input pc, [B, K, 3]

        Returns:
            torch.FloatTensor: pc after outlier removal, [B, N, 3]
        """
        pc = x.clone().detach().double()
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)  # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # [B, K]
        mean = torch.mean(value, dim=-1)  # [B]
        std = torch.std(value, dim=-1)  # [B]
        threshold = mean + self.alpha * std  # [B]
        bool_mask = (value <= threshold[:, None])  # [B, K]
        sel_pc = x[0][bool_mask[0]].unsqueeze(0)
        sel_pc = self.process_data(sel_pc)
        for i in range(1, B):
            proc_pc = x[i][bool_mask[i]].unsqueeze(0)
            proc_pc = self.process_data(proc_pc)
            sel_pc = torch.cat([sel_pc, proc_pc], dim=0)
        return sel_pc

    def process_data(self, pc, npoint=None):
        """Process point cloud data to be suitable for
            PU-Net input.
        We do two things:
            sample npoint or duplicate to npoint.

        Args:
            pc (torch.FloatTensor): list input, [(N_i, 3)] from SOR.
                Need to pad or trim to [B, self.npoint, 3].
        """
        if npoint is None:
            npoint = self.npoint
        proc_pc = pc.clone()
        num = npoint // pc.size(1)
        for _ in range(num-1):
            proc_pc = torch.cat([proc_pc, pc], dim=1)
        num = npoint - proc_pc.size(1)
        duplicated_pc = proc_pc[:, :num, :]
        proc_pc = torch.cat([proc_pc, duplicated_pc], dim=1)
        assert proc_pc.size(1) == npoint
        return proc_pc

    def forward(self, x):
        with torch.enable_grad():
            x = self.outlier_removal(x)
            x = self.process_data(x)  # to batch input
        return x


class SRSDefense(nn.Module):
    """Random dropping points as defense.
    """

    def __init__(self, drop_num=500):
        """SRS defense method.

        Args:
            drop_num (int, optional): number of points to drop.
                                        Defaults to 500.
        """
        super(SRSDefense, self).__init__()

        self.drop_num = drop_num

    def random_drop(self, pc):
        """Random drop self.drop_num points in each pc.

        Args:
            pc (torch.FloatTensor): batch input pc, [B, K, 3]
        """
        B, K = pc.shape[:2]
        idx = [np.random.choice(K, K - self.drop_num, replace=False) for _ in range(B)]
        pc = torch.stack([pc[i][torch.from_numpy(idx[i]).long().to(pc.device)] for i in range(B)])
        return pc

    def forward(self, x):
        with torch.no_grad():
            x = self.random_drop(x)
        return x




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count