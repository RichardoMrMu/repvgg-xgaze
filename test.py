# -*- coding: utf-8 -*-
# @Time    : 2021/4/28 16:45
# @Author  : RichardoMu
# @File    : test.py
# @Software: PyCharm
import yacs.config
import os
import numpy as np
import h5py
import random
from models import get_RepVGG_func_by_name
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import List
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class GazeDataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None,
                 is_shuffle=True,
                 index_file=None, is_load_label=True):
        self.path = dataset_path

        self.sub_folder = sub_folder
        self.is_load_label = is_load_label

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0


        if index_file:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file).astype(np.int)

        if is_shuffle:
            random.shuffle(self.idx_to_kv[:])  # random the order to stable the training

        self.hdf = None
        self.transform = transform

    def __len__(self):
        return len(self.idx_to_kv)


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



def get_index_file(train_files, is_shuffle=True):
    if not os.path.exists('./test_index_file'):
        os.mkdir('./test_index_file')
    if os.path.exists('./test_index_file')  and os.path.exists("./test_index_file/index_file_test.txt"):
        return
    # 保存
    hdfs = {}
    # 读取hdfs文件
    for num_i in range(len(train_files)):
        hdfs[num_i] = h5py.File(train_files[num_i],'r',swmr=True)
        assert hdfs[num_i].swmr_mode

    # 读取文件中的shape，构建映射 (num_i,x) num_i为第num_i个h5文件，x为该文件的第x个图片，其中每个h5文件有n个图片
    idx_to_kv = []
    for num_i in range(len(train_files)):
        n = hdfs[num_i]['face_patch'].shape[0]
        idx_to_kv += [(num_i,i) for i in range(n)]

    for num_i in range(len(train_files)):
        if hdfs[num_i]:
            hdfs[num_i].close()
            hdfs[num_i] = None
    if is_shuffle:
        random.shuffle(idx_to_kv)
    idx_to_kv = np.array(idx_to_kv).astype(np.int)
    print(type(idx_to_kv))
    np.savetxt("./test_index_file/index_file_test.txt", idx_to_kv)


def get_data_loader():
    # load dataset
    refer_list_file = os.path.join('/home/data/tbw_data/xgaze/','test')
    train_data_dir = os.listdir(refer_list_file)
    train_files = [os.path.join(refer_list_file,train_data_dir[i]) for i in range(len(train_data_dir))]
    get_index_file(train_files,is_shuffle=False)

    test_set = GazeDataset(dataset_path='/home/data/tbw_data/xgaze/', keys_to_use=train_data_dir,sub_folder='test',
                            transform=trans, is_shuffle=False, index_file='./test_index_file/index_file_test.txt',
                            is_load_label=False)

    test_loader = DataLoader(test_set,batch_size=256,num_workers=20)
    return test_loader


def validate(model,test_loader):
    model.eval()
    print(len(test_loader))
    pred_gaze_list = np.zeros((len(test_loader),2))
    save_index = 0
    with torch.no_grad():
        for step, (input_var) in enumerate(test_loader):
            print(step)
            input_var = torch.autograd.Variable(input_var.float().cuda())
            # test gaze net
            pred_gaze = model(input_var)
            pred_gaze_list[save_index:save_index+256,:] = pred_gaze.cpu().data.numpy()
            save_index += input_var.size(0)

    print(pred_gaze_list.shape)
    np.savetxt("./test_index_file/within_eva_results.txt",pred_gaze_list,delimiter=',')
    return

def main():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1234)
    np.random.seed(1234)
    repvgg_build_func = get_RepVGG_func_by_name("RepVGG-D2se")
    model = repvgg_build_func(deploy=False)
    checkpoints = torch.load('/home/tbw/work/code/eth-compitition/ckpt/repvgg_d2se/exp00/RepVGG-D2se_epoch_6_ckpt.pth.tar')
    # if 'model_state' in checkpoints:
    checkpoints = checkpoints['model_state']
    # ckpt = {k.replace('module.', ''): v for k, v in checkpoints.items()}  # strip the names
    model.load_state_dict(checkpoints, strict=True)
    model = model.cuda()
    test_loader = get_data_loader()
    validate(model,test_loader)

if __name__ == '__main__':
    main()
