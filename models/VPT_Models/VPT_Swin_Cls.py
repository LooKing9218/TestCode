import torch

import torch.nn as nn
from VPT_Models.mmseg_un.models.backbones import SwinTransformer_Cls


class swin_network(nn.Module):
    '''
                end to end net
    '''
    def __init__(self,args,num_classes=9, input_size=(224,224)):
        super(swin_network, self).__init__()
        self.args = args

        self.backbone = SwinTransformer_Cls(
             args=args,
             pretrain_img_size=input_size, # related to ape (True)
             patch_size=4,
             in_chans=3,
             embed_dim=128,
             depths= [2, 2, 18, 2], # [2,2], #[2, 2, 18, 2],
             num_heads=[4, 8, 16, 32], #[4,8], #[4, 8, 16, 32],
             window_size=7,
             mlp_ratio=4.,
             qkv_bias=True,
             qk_scale=None,
             drop_rate=0.,
             attn_drop_rate=0.,
             drop_path_rate=0.3,
             norm_layer=nn.LayerNorm,
             ape=False,
             patch_norm=True,
             out_indices= (0, 1, 2, 3), #(0,1), #(0, 1, 2, 3),
             use_checkpoint=False,
            device=args.device
        )


        # UNet up-root

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(512, num_classes)
        self.init_weights()


    def init_weights(self):
        #import pdb;pdb.set_trace()
        pretrain_ckpt = torch.load(self.args.snapshot_fname)
        pretrain_dict = pretrain_ckpt['state_dict']

        cnt = 0
        state_dict = self.backbone.state_dict()
        for key_old in pretrain_dict.keys():
            key = key_old[9:]#'backbone.'+ key_old
            if key not in state_dict:
                continue
            value = pretrain_dict[key_old]
            if not isinstance(value, torch.FloatTensor):
                value = value.data
            state_dict[key] = value
            cnt+=1
        print('Loaded para num: ',cnt)
        self.backbone.load_state_dict(state_dict)

    def forward(self, x):
        x = self.backbone(x)
        # print("len(x) == {}, x[-1].shape == {}".format(len(x),x[-1].shape))
        # print("x.shape === {}".format(x.shape))
        x = self.avgpool(x[-1])
        # print("x.shape === {}".format(x.shape))
        x = torch.flatten(x, 1)
        # print("x.shape === {}".format(x.shape))
        x = self.fc1(x)
        # print("x.shape === {}".format(x.shape))
        x = self.fc2(x)

        return x

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):
        return torch.cat((upsampled, bypass), 1)



# if __name__ == '__main__':
#     from utils.config_Multimodal import DefaultConfig
#     args = DefaultConfig()  # 配置设置
#     input_size = [224,224]
#
#     net = swin_network(args,input_size=input_size).cuda()
#     x = torch.rand((2, 3, input_size[0], input_size[1])).cuda()
#     forward = net.forward(x)
#     print(forward.shape)
#     print(type(forward))
