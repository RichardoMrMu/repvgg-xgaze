# -*- coding: utf-8 -*-
# @Time    : 2020-12-07 10:25
# @Author  : RichardoMu
# @File    : data_loader_tbw.py
# @Software: PyCharm


import numpy as np
import h5py
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import random
from typing import List
import yacs.config
trans_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# def get_data_loader(data_dir:str,
#                     batch_size = 50,
#                     dataset_split=[0.85,0.1,0.05],
#                     num_workers = 5,
#                     is_shuffle=True):
def get_data_loader(config: yacs.config.CfgNode):
    # load dataset
    refer_list_file = os.path.join(config.dataset.dataset_dir,'train_au')
    train_data_dir = os.listdir(refer_list_file)
    train_files = [os.path.join(refer_list_file,train_data_dir[i]) for i in range(len(train_data_dir))]
    get_index_file(train_files,config.train.dataset_split,is_shuffle=config.train.shuffle)
    train_set = GazeDataset(dataset_path=config.dataset.dataset_dir,keys_to_use=train_data_dir,sub_folder='train_au',
                            transform=trans,is_shuffle=config.train.shuffle,index_file='./index_file/index_file_train.txt',is_load_label=True)
    val_set = GazeDataset(dataset_path=config.dataset.dataset_dir, keys_to_use=train_data_dir,sub_folder='train_au',
                            transform=trans, is_shuffle=config.train.shuffle, index_file='./index_file/index_file_val.txt',
                            is_load_label=True)
    test_set = GazeDataset(dataset_path=config.dataset.dataset_dir, keys_to_use=train_data_dir,sub_folder='train_au',
                            transform=trans, is_shuffle=False, index_file='./index_file/index_file_test.txt',
                            is_load_label=True)
    train_loader = DataLoader(train_set,batch_size=config.train.batch_size,num_workers=config.train.train_dataloader.num_workers,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_set,batch_size=config.val.batch_size,num_workers=config.val.val_dataloader.num_workers,pin_memory=True,drop_last=True)
    test_loader = DataLoader(test_set,batch_size=config.test.batch_size,num_workers=config.test.test_dataloader.num_workers,pin_memory=True,drop_last=True)
    return train_loader,val_loader,test_loader


def get_index_file(train_files, dataset_split, is_shuffle=True):


    if os.path.exists('./index_file') and os.path.exists('./index_file/index_file_train.txt') and os.path.exists(
            "./index_file/index_file_test.txt") and os.path.exists("./index_file/index_file_val.txt"):
        return

    # 保存
    hdfs = {}
    # 读取hdfs文件
    for num_i in range(len(train_files)):
        hdfs[num_i] = h5py.File(train_files[num_i],'r',swmr=True)
        assert hdfs[num_i].swmr_mode

    # 读取文件中的shape，构建映�?(num_i,x) num_i为第num_i个h5文件，x为该文件的第x个图片，其中每个h5文件有n个图�?
    idx_to_kv = []
    for num_i in range(len(train_files)):
        n = hdfs[num_i]['face_patch'].shape[0]
        idx_to_kv += [(num_i,i) for i in range(n)]
    # print(idx_to_kv)
    for num_i in range(len(train_files)):
        if hdfs[num_i]:
            hdfs[num_i].close()
            hdfs[num_i] = None
    if is_shuffle:
        random.shuffle(idx_to_kv)
    idx_to_kv = np.array(idx_to_kv).astype(np.int)
    train_idx_to_kv = idx_to_kv[:int(len(idx_to_kv)*dataset_split[0])]
    val_idx_to_kv = idx_to_kv[int(len(idx_to_kv)*(dataset_split[0])):
                             int(len(idx_to_kv)*(dataset_split[0]+dataset_split[1]))]
    test_idx_to_kv = idx_to_kv[int(len(idx_to_kv)*(dataset_split[0]+dataset_split[1])):]
    # train_idx_to_kv = idx_to_kv[:600]
    # val_idx_to_kv = idx_to_kv[:600]
    # test_idx_to_kv = idx_to_kv[:600]
    if not os.path.exists('./index_file'):
        os.mkdir('./index_file')
    np.savetxt("./index_file/index_file_train.txt",train_idx_to_kv)
    np.savetxt("./index_file/index_file_val.txt",val_idx_to_kv)
    np.savetxt("./index_file/index_file_test.txt", test_idx_to_kv)


class GazeDataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None,
                 is_shuffle=True,
                 index_file=None, is_load_label=True):
        self.path = dataset_path
        # self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0

        # for num_i in range(0, len(self.selected_keys)):
        #     file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
        #     self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
        #     # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
        #     assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        # if index_file is None:
            # self.idx_to_kv = []
            # for num_i in range(0, len(self.selected_keys)):
            #     n = self.hdfs[num_i]["face_patch"].shape[0]
            #     self.idx_to_kv += [(num_i, i) for i in range(n)]
        # else:
        if index_file:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file).astype(np.int)
        # # 将读取的hdfs文件关闭
        # for num_i in range(0, len(self.hdfs)):
        #     if self.hdfs[num_i]:
        #         self.hdfs[num_i].close()
        #         self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.transform = transform

    def __len__(self):
        return len(self.idx_to_kv)

    # def __del__(self):
    #     for num_i in range(0, len(self.hdfs)):
    #         if self.hdfs[num_i]:
    #             self.hdfs[num_i].close()
    #             self.hdfs[num_i] = None

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        # Get face image
        image = self.hdf['face_patch'][idx, :]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = self.transform(image)

        # Get labels
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :]
            gaze_label = gaze_label.astype('float')
            return image, gaze_label
        else:
            return image


