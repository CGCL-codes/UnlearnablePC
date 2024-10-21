import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

# 将KITTI数据集中场景和车辆进行分割，我们对车辆进行操作

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class KITTIDataLoader(Dataset):
    def __init__(self, root,  npoints=256, split='train', train_num = 1000, uniform=False, cache_size=15000):
        self.root = root
        self.npoints = npoints
        self.uniform = uniform
        file_names = os.listdir(self.root)

        #! random choice 2662
        assert (split == 'train' or split == 'test')
        labels = []
        file_paths = []
        v_counter = 0
        h_counter = 1
        file_names = np.random.permutation(file_names)
        for file in file_names:
            if (file.split('-')[1] == 'vehicle'):
                if v_counter < 1331:
                    file_paths.append(file)
                    labels.append(1)
                    v_counter += 1
            else:
                if h_counter < 1331:
                   file_paths.append(file)
                   labels.append(0)
                   h_counter +=1

        if (split == 'train'):
            labels = labels[:train_num]
            file_paths = file_paths[:train_num]
        else:
            labels = labels[train_num:]
            file_paths = file_paths[train_num:train_num+662]
        # list of (shape_name, shape_txt_file_path) tuple

        self.datapath = [(labels[i], os.path.join(self.root, file_paths[i])) for i in range(len(file_paths))]
        print('The size of %s data is %d'%(split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple
        

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = fn[0]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.load(fn[1]).astype(np.float32)[:, 0:3]
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)




if __name__ == '__main__':
    import torch
    
    data = KITTIDataLoader('../../DATA/KITTI/training/object_cloud',split='train', uniform=False)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)