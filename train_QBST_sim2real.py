import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from collections import Counter
from itertools import groupby
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, ConcatDataset
from utils.pc_utils import random_rotate_one_axis
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
from data.dataloader_sim2real import ScanNet, ScanNet_xyz, ModelNet, Sim2Real, label_to_idx, NUM_POINTS
from Models_Norm import PointNet, DGCNN

NWORKERS = 4
MAX_LOSS = 9 * (10 ** 9)


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--exp_name', type=str, default='Sim2Real_fintue', help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--dataroot', type=str, default='./data', metavar='N', help='data path')
parser.add_argument('--model_file', type=str, default='model.ptdgcnn', help='pretrained model file')
parser.add_argument('--src_dataset', type=str, default='sim2real', choices=['modelnet', 'sim2real', 'scannet', 'scannet_single'])
parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'sim2real', 'scannet', 'scannet_single'])
parser.add_argument('--epochs', type=int, default=10, help='number of episode to train')
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_voxels', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--DefRec_on_src', type=str2bool, default=False, help='Using DefRec in source')
parser.add_argument('--DefRec_on_trgt', type=str2bool, default=False, help='Using DefRec in target')
parser.add_argument('--DefCls_on_src', type=str2bool, default=False, help='Using DefCls in source')
parser.add_argument('--DefCls_on_trgt', type=str2bool, default=False, help='Using DefCls in target')
parser.add_argument('--PosReg_on_src', type=str2bool, default=False, help='Using PosReg in source')
parser.add_argument('--PosReg_on_trgt', type=str2bool, default=False, help='Using PosReg in target')
parser.add_argument('--apply_PCM', type=str2bool, default=False, help='Using mixup in source')
parser.add_argument('--apply_GRL', type=str2bool, default=False, help='Using gradient reverse layer')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--cls_weight', type=float, default=0.5, help='weight of the classification loss')
parser.add_argument('--grl_weight', type=float, default=0.5, help='weight of the GRL loss')
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--DefCls_weight', type=float, default=0.5, help='weight of the DefCls loss')
parser.add_argument('--PosReg_weight', type=float, default=0.5, help='weight of the PosReg loss')
parser.add_argument('--output_pts', type=int, default=512, help='number of decoder points')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--iteration', type=float, default=10, help='iteration number for self-training')
parser.add_argument('--threshold', type=float, default=0.8, help='confidence threshold')
parser.add_argument('--epsilon', type=float, default=0.005, help='constant')

args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

random.seed(1)
# np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')

# ==================
# Init Model
# ==================
if args.model == 'pointnet':
    model = PointNet(args)
    new_model = PointNet(args)
    model.load_state_dict(torch.load('./experiments/Sim2Real/model.ptpointnet'))
elif args.model == 'dgcnn':
    model = DGCNN(args)
    new_model = DGCNN(args)
    model.load_state_dict(torch.load('./experiments/' + args.model_file))
else:
    raise Exception("Not implemented")

model = model.to(device)
new_model = new_model.to(device)

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)

# ==================
# loss function
# ==================
opt = optim.SGD(new_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam(new_model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(opt, args.epochs)
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch

# ==================
# Read Data
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


trgt_dataset = args.trgt_dataset
src_dataset = args.src_dataset
data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'sim2real': Sim2Real, 'scannet_single': ScanNet_xyz}
src_trainset = data_func[src_dataset](io, args.dataroot, 'train')
trgt_trainset = data_func[trgt_dataset](io, args.dataroot, 'train')
src_testset = data_func[src_dataset](io, args.dataroot, 'test')
trgt_testset = data_func[trgt_dataset](io, args.dataroot, 'test')
src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
trgt_train_sampler, trgt_valid_sampler = split_set(trgt_trainset, trgt_dataset, "target")

# dataloaders for finetue and test
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                              sampler=src_train_sampler, drop_last=True)
trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                               sampler=trgt_train_sampler, drop_last=True)
src_test_loader = DataLoader(src_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)


# ==================
# utils
# ==================
def entropy(*c):
    result = -1
    if len(c) > 0:
        result = 0
    for x in c:
        result += (-x) * math.log(x, 2)
    return result


def kld_loss(pred):
    pred_conf = torch.max(pred, 1)[0]
    vaild = pred_conf >= threshold
    vaild = vaild.reshape(-1, 1).float()
    logsoftmax = F.log_softmax(pred, dim=1)   # compute the log of softmax values
    kld = torch.sum(-logsoftmax * vaild / 10)
    return kld / vaild.sum()


# ==================
# select_target_data
# ==================
def select_target_by_conf(trgt_train_loader, model=None):
    pc_list = []
    label_list = []
    sfm = nn.Softmax(dim=1)

    with torch.no_grad():
        model.eval()
        for data in trgt_train_loader:
            data = data[0].to(device)
            data = data.permute(0, 2, 1)

            logits = model(data, activate_DefRec=False)
            cls_conf = sfm(logits['cls'])
            mask = torch.max(cls_conf, 1)  # 2 * b
            index = 0
            for i in mask[0]:
                if i > threshold:
                    pc_list.append(data[index].cpu().numpy())
                    label_list.append(mask[1][index].cpu().numpy())
                index += 1
    return np.array(pc_list), np.array(label_list)


def gather(seq, func):
    return [list(g) for k, g in groupby(sorted(seq, key=func), key=func)]


def select_target_by_clswise_conf(trgt_train_loader, model=None):
    pc_list = []
    pc_select_list = []
    label_select_list = []
    mask_val_list = []
    mask_ind_list = []
    sfm = nn.Softmax(dim=1)

    with torch.no_grad():
        model.eval()
        for data in trgt_train_loader:
            data = data[0].to(device)
            data = data.permute(0, 2, 1)

            logits = model(data, activate_DefRec=False)
            cls_conf = sfm(logits['cls'])
            mask = torch.max(cls_conf, 1)  # 2 * b

            mask_val_list.extend(mask[0].cpu().numpy())  # confidence
            mask_ind_list.extend(mask[1].cpu().numpy())  # pseudo label

            pc_list.append(data.cpu().numpy().tolist())

        mask_val_arr = np.array(mask_val_list)
        mask_ind_arr = np.array(mask_ind_list)
        pc_arr = np.array(sum(pc_list, []))
        mask_selct_by_threshold = mask_val_arr >= threshold

        mask_val_arr = mask_val_arr[mask_selct_by_threshold]
        mask_ind_arr = mask_ind_arr[mask_selct_by_threshold]
        pc_arr = pc_arr[mask_selct_by_threshold]

        sorted_id = np.argsort(mask_ind_arr)
        clswise_count = dict(sorted(Counter(mask_ind_arr).items(), key=operator.itemgetter(0), reverse=False))
        print(clswise_count)
        count_sum = sum(clswise_count.values())
        print(count_sum)
        start = 0
        for k, v in clswise_count.items():
            sorted_id_slice = sorted_id[start: start + v]
            mu = 1.0 - (v / count_sum)
            print(mu)
            lenth = math.ceil(len(sorted_id_slice) * mu)

            mask_val_slice = mask_val_arr[sorted_id_slice]
            mask_val_select_slice_sorted_id = np.argsort(mask_val_slice)[:lenth]  # sorted by confidence

            pc_list_slice = pc_arr[sorted_id_slice]
            pc_list_select_slice = pc_list_slice[mask_val_select_slice_sorted_id]
            pc_select_list.append(pc_list_select_slice.tolist())

            mask_ind_slice = mask_ind_arr[sorted_id_slice]
            label_list_select_slice = mask_ind_slice[mask_val_select_slice_sorted_id]
            label_select_list.append(label_list_select_slice.tolist())

            start += v

        pc_select_list = sum(pc_select_list, [])
        label_select_list = sum(label_select_list, [])
        print(len(pc_select_list))
        print(len(label_select_list))

    return np.array(pc_select_list), np.array(label_select_list)


class DataLoad(Dataset):
    def __init__(self, io, data, partition='train'):
        self.partition = partition
        self.pc, self.label = data
        self.num_examples = len(self.pc)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in trgt_new_dataset: " + str(len(self.pc)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in trgt_new_dataset " + partition + " set: " +
                  str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.copy(self.pc[item])
        pointcloud = random_rotate_one_axis(pointcloud.transpose(1, 0), "z")
        # pointcloud = pointcloud.transpose(1, 0)
        label = np.copy(self.label[item])
        return (pointcloud, label)

    def __len__(self):
        return len(self.pc)


def self_train(train_loader, src_train_loader, model):
    count = 0.0
    print_losses = {'cls': 0.0}
    for epoch in range(args.epochs):
        model.train()
        for data1, data2 in zip(train_loader, src_train_loader):
            opt.zero_grad()
            batch_size = data1[1].size()[0]
            t_data, t_labels = data1[0].to(device), data1[1].to(device)
            t_data = t_data.permute(0, 2, 1)
            t_logits = model(t_data, activate_DefRec=False)
            # s_data, s_labels = data2[0].to(device), data2[1].to(device)
            # s_data = s_data.permute(0, 2, 1)
            # s_logits = model(s_data, activate_DefRec=False)
            # loss = criterion(t_logits["cls"], t_labels) + criterion(s_logits["cls"], s_labels)
            # loss = criterion(t_logits["cls"], t_labels) + kld_loss(t_logits["cls"]) * 0.5
            loss = criterion(t_logits["cls"], t_labels)
            print_losses['cls'] += loss.item() * batch_size
            loss.backward()
            count += batch_size
            opt.step()
        scheduler.step()

        print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
        io.print_progress("Target_new", "Trn", epoch, print_losses)


# ==================
# Validation/test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):
    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            logits = model(data, activate_DefRec=False)
            loss = criterion(logits["cls"], labels)
            print_losses['cls'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits["cls"].max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return test_acc, print_losses['cls'], conf_mat


trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, model, "Target", "Test", 0)
io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_test_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))
threshold = args.threshold
if trgt_test_acc > 0.9:
    threshold = 0.98
trgt_new_best_val_acc = 0
for i in range(args.iteration):
    print(threshold)
    model = copy.deepcopy(best_model)
    trgt_select_data = select_target_by_clswise_conf(trgt_train_loader, model)
    trgt_new_data = DataLoad(io, trgt_select_data)
    trgt_new_train_sampler, trgt_new_valid_sampler = split_set(trgt_new_data, trgt_dataset, "target_new")
    train_loader = DataLoader(trgt_new_data, num_workers=NWORKERS, batch_size=args.batch_size,
                              sampler=trgt_new_train_sampler, drop_last=True)
    test_loader = DataLoader(trgt_new_data, num_workers=NWORKERS, batch_size=args.batch_size,
                             sampler=trgt_new_valid_sampler, drop_last=True)
    self_train(train_loader, src_train_loader, new_model)
    trgt_new_val_acc, _, _ = test(test_loader, new_model, "Target_New", "Val", 0)
    #trgt_new_val_acc, _, _ = test(src_test_loader, model, "Source", "Val", 0)
    test(trgt_test_loader, new_model, "Target", "Test", 0)
    if trgt_new_val_acc > trgt_new_best_val_acc:
        trgt_new_best_val_acc = trgt_new_val_acc
        best_model = io.save_model(new_model)
    threshold += args.epsilon

trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, best_model, "Target", "Test", 0)
io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_test_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))
