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

class MultiModalDataset(Dataset):
    def __init__(self,root,txt_file,mode = 'train',height=420,width=420):
        print("=========== Multi-modality ===========")
        self.data_list = self.get_files(root,txt_file=txt_file)
        if mode == 'train':
            self.transforms= T.Compose([
                T.Resize((height,width)),
                T.RandomResizedCrop(size=410,scale=(0.25,1)),
                # T.CenterCrop(size=384),
                T.RandomHorizontalFlip(),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])
        else:

            self.transforms = T.Compose([
                T.Resize((height,width)),
                T.RandomResizedCrop(size=410,scale=(0.25,1)),
                # T.CenterCrop(size=384),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])
        print("self.transforms == {}".format(self.transforms))

    def get_files(self,root, txt_file):
        img_list = []
        with open(os.path.join(txt_file), 'r') as FF:
            content_lines = FF.readlines()
            for line in content_lines:
                line_list = line.split(",")
                img_list.append(
                    [
                       [os.path.join(root, line_list[0], "Fundus_resize", line_list[1])]+glob.glob(os.path.join(root, "{}/OCT/Square".format(line_list[0])) + "/*.tif"),
                        int(line_list[2]),
                        float(line_list[3])
                    ]
                )

        return img_list

    def __getitem__(self,index):
        return_list = []
        image_list,valabel,revalabel = self.data_list[index]
        for image_file in image_list:
            img = Image.open(image_file)
            img = img.convert("RGB")
            img = self.transforms(img)
            return_list.append(img)

        return return_list,valabel,revalabel

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
