import os
import glob
from plyfile import PlyData
import h5py
import numpy as np
from torch.utils.data import Dataset
from utils.pc_utils import (farthest_point_sample_np, scale_to_unit_cube, jitter_pointcloud,
                            rotate_shape, random_rotate_one_axis)

eps = 10e-4
NUM_POINTS = 1024
idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                4: "chair", 5: "lamp", 6: "monitor",
                7: "plant", 8: "sofa", 9: "table"}
label_to_idx = {"bathtub": 0, "bed": 1, "bookshelf": 2, "cabinet": 3,
                "chair": 4, "lamp": 5, "monitor": 6,
                "plant": 7, "sofa": 8, "table": 9}


def read_ply(filename):
    """ read cordinates, return n * 3 """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


class ScanNet_ply(Dataset):
    """
    scannet dataset for pytorch dataloader
    """

    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation
        self.pc_list = []
        self.lbl_list = []
        DATA_DIR = os.path.join(dataroot, "DA_data", "scannet")

        ply_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.ply')))

        for _dir in ply_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-2]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in scannet : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in scannet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = read_ply(self.pc_list[item]).astype(np.float32)
        label = np.copy(self.label[item])
        pointcloud = scale_to_unit_cube(pointcloud)
        # Rotate ScanNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples

        if self.random_rotation == True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)

        return (pointcloud, label)

    def __len__(self):
        return len(self.pc_list)

    # scannet is rotated such that the up direction is the z axis
    def rotate_pc(self, pointcloud):
        pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud


class ScanNet_xyz(Dataset):
    """
    Scannet_single_obj dataset for pytorch dataloader
    """

    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation
        self.pc_list = []
        self.lbl_list = []
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "Scannet_single_obj_1012_new")

        ply_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.xyz')))

        for _dir in ply_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in scannet : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in scannet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.loadtxt(self.pc_list[item]).astype(np.float32)
        label = np.copy(self.label[item])
        pointcloud = scale_to_unit_cube(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples

        if self.random_rotation == True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)

        return (pointcloud, label)

    def __len__(self):
        return len(self.pc_list)


def load_data_h5py_scannet10(partition, dataroot):
    """
    Input:
        partition - train/test
    Return:
        data,label arrays
    """
    DATA_DIR = dataroot + '/PointDA_data/scannet'
    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(DATA_DIR, '%s_*.h5' % partition))):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:]
        label = f['label'][:]
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return np.array(all_data).astype('float32'), np.array(all_label).astype('int64')


class ScanNet(Dataset):
    """
    scannet dataset for pytorch dataloader
    """

    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation

        # read data
        self.data, self.label = load_data_h5py_scannet10(self.partition, dataroot)
        self.num_examples = self.data.shape[0]

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in scannet" + ": " + str(self.data.shape[0]))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in scannet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.copy(self.data[item])[:, :3]
        label = np.copy(self.label[item])
        pointcloud = scale_to_unit_cube(pointcloud)
        # Rotate ScanNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

            # apply data rotation and augmentation on train samples

        if self.random_rotation == True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)

        return (pointcloud, label)

    def __len__(self):
        return self.data.shape[0]

    # scannet is rotated such that the up direction is the z axis
    def rotate_pc(self, pointcloud):
        pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud


class Sim2Real(Dataset):
    """
    sim2real dataset for pytorch dataloader
    """

    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation
        self.pc_list = []
        self.lbl_list = []
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "sim2real", "raw_pc_multiview_point_at")

        ply_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.xyz')))

        for _dir in ply_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in sim2real : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in sim2real " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        # pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        pointcloud = np.loadtxt(self.pc_list[item]).astype(np.float32)
        label = np.copy(self.label[item])
        pointcloud = scale_to_unit_cube(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.random_rotation == True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)
        return (pointcloud, label)

    def __len__(self):
        return len(self.pc_list)


class ModelNet(Dataset):
    """
    modelnet dataset for pytorch dataloader
    """

    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation
        self.pc_list = []
        self.lbl_list = []
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "modelnet")

        npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.npy')))

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in modelnet : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        label = np.copy(self.label[item])
        pointcloud = scale_to_unit_cube(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.random_rotation == True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)

        return (pointcloud, label)

    def __len__(self):
        return len(self.pc_list)
