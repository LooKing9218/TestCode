# --coding:utf-8--
from torch.utils.data import Dataset
from torchvision import transforms as T 
from PIL import Image
import pandas as pd
import os 
import glob
import torch
#1.set random seed
import torch._utils
from itertools import islice

class DatasetCFP(Dataset):
    def __init__(self,root,data_file,mode = 'train'):
        self.data_list = self.get_files(root,data_file=data_file)
        if mode == 'train':
            self.transforms= T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])
        else:

            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])

    def get_files(self,root, data_file):
        import csv
        csv_reader = csv.reader(open(data_file))
        img_list = []
        for line in islice(csv_reader, 1, None):
            img_list.append(
                [
                   os.path.join(root,line[0]),
                    int(line[1])
                ]
            )
        return img_list

    def __getitem__(self,index):
        image_file,label = self.data_list[index]
        img = Image.open(image_file).convert("RGB")
        img_tensor = self.transforms(img)

        return img_tensor, label

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":

    from torch.utils.data import DataLoader

    train_loader = DataLoader(MultiModalDataset(
        root="/home/wang_meng/DRCRNetDataset/Master_Set",
        mode = 'train',
        txt_file="/home/wang_meng/Project/MultiModal_Project/MultiModalVAPro/data_txt/train_file.txt"),
                              batch_size=2,shuffle=True,pin_memory=True)
    val_loader = DataLoader(MultiModalDataset(
        root="/home/wang_meng/DRCRNetDataset/Master_Set",
        mode = 'val',
        txt_file="/home/wang_meng/Project/MultiModal_Project/MultiModalVAPro/data_txt/val_file.txt"),
                              batch_size=2,shuffle=True,pin_memory=True)
    test_loader = DataLoader(MultiModalDataset(
        root="/home/wang_meng/DRCRNetDataset/Master_Set",
        mode = 'test',
        txt_file="/home/wang_meng/Project/MultiModal_Project/MultiModalVAPro/data_txt/test_file.txt"
    ),
        batch_size=2,shuffle=False,pin_memory=True)
    import time
    star = time.time()

    print("train_loader == {}".format(train_loader.__len__()))
    print("val_loader == {}".format(val_loader.__len__()))
    print("test_loader == {}".format(test_loader.__len__()))
    for idx,img_data in enumerate(test_loader):
        print("idx == {}, img_data.shape == {},  valabel == {}, revalabel == {}".format(
            idx,img_data[0][0].shape,img_data[1],img_data[2]
        ))
    end = time.time()
    print("Times == {}".format(end-star))
