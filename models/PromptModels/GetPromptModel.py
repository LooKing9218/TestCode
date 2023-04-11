"""
build_promptmodel script  ver： Mar 25th 19:20

"""
import timm
import torch
from .structure import *
from VPT_Models.VPT_Swin_Cls import swin_network as vpt_swin


def build_promptmodel(args):
    print("=================== Swin ===================")
    model = vpt_swin(args,num_classes=args.num_classes, input_size=(256,256)).to(args.device)

    return model


# if __name__ == "__main__":
#     model = vpt_swin(args, input_size=args.input_size)
#
#     try:
#         img = torch.randn(1, 3, args.input_size[0], args.input_size[1])
#         preds = model(img)  # (1, class_number)
#         print('test model output：', preds)
#     except:
#         print("Problem exist in the model defining process！！")
#         return -1
#     else:
#         print('model is ready now!')
#     print(model)
#     return model