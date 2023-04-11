import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict

class ResNet(nn.Module):
    def __init__(self,num_classes=1):
        super(ResNet, self).__init__()
        resnet_img = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-34
        modules_img = list(resnet_img.children())[:-2]
        self.resnet_img = nn.Sequential(*modules_img)

        self.avgpool_fun = nn.AdaptiveAvgPool2d((1,1))  #
        self.affine_classifier = nn.Linear(2048, num_classes)


    def forward(self, image):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out_img = self.resnet_img(image)
        avg_feature = self.avgpool_fun(out_img)
        avg_feature = torch.flatten(avg_feature, 1)
        result = self.affine_classifier(avg_feature)
        return result

if __name__ == '__main__':
    net = ResNet(num_classes=1)
    images = torch.rand(2, 3, 512, 512)
    output = net(images)
    print(output.shape)
