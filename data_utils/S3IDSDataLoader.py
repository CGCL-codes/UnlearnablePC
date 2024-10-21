import numpy as np
import h5py
import os
from torch.utils.data import Dataset


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label

def get_current_data_h5(pcs, labels, num_points):
	#shuffle points to sample
	idx_pts = np.arange(pcs.shape[1])
	np.random.shuffle(idx_pts)

	sampled = pcs[:,idx_pts[:num_points],:]
	#sampled = pcs[:,:num_points,:]

	#shuffle point clouds per epoch
	idx = np.arange(len(labels))
	np.random.shuffle(idx)

	sampled = sampled[idx]
	labels = labels[idx]

	return sampled, labels


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


class S3IDSDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', test_area:str = '6', uniform=False, cache_size=5000):
        self.root = root
        self.npoint = npoint
        self.uniform = uniform
        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        #! random choice 2662
        assert (split == 'train' or split == 'test')
        files = [line.rstrip() for line in open(root + '/all_files.txt')]
        room_filelist = [line.rstrip() for line in open(root + '/room_filelist.txt')]

        idxs = []
        if split == 'train':
            for i, name in enumerate(room_filelist):
                if test_area not in name:
                    idxs.append(i)
        else:
            for i, name in enumerate(room_filelist):
                if test_area in name:
                    idxs.append(i)
        # Load ALL data
        data_batch_list = []
        label_batch_list = []
        for h5_filename in files:
            data_batch, label_batch = load_h5(os.path.join(root, h5_filename.split('/')[-1]))
            data_batch_list.append(data_batch)
            label_batch_list.append(label_batch)
        data_batches = np.concatenate(data_batch_list, 0)
        label_batches = np.concatenate(label_batch_list, 0)
        print(data_batches.shape)
        print(label_batches.shape)  

        self.data = data_batches[idxs,...]
        self.labels = data_batches[idxs]

        print('The size of %s data is %d'%(split,len(self.data)))

    def __len__(self):
        return len(self.data)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            point_set = self.data[index].astype(np.int32)
            cls = self.labels[index].astype(np.int32)

            point_set = point_set[0:self.npoint,:]
            cls = cls[0:self.npoint]
            # point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

import  matplotlib.pyplot as plt
def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='GnBu', zdir='y',
                        xlim=(-0.9, 0.9), ylim=(-0.9, 0.9), zlim=(-0.9, 0.9)):
    if sizes is None:
        sizes = [0.35 for _ in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[i])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


if __name__ == '__main__':
    import torch
    # 15 classes
    data = S3IDSDataLoader('data/indoor3d_sem_seg_hdf5_data',split='train', uniform=False)  # 2309, 581
    DataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)
        # plot_pcd_three_views('ttt.png', [point[0]], ['a', 'b', 'c'])
        break
