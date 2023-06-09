import torch
import torchvision
import torch.nn as nn

class ResUnNet(nn.Module):
    def __init__(self,num_classes=1):
        print("=================== ResUnNetFC1 NoDrop ===================")
        super(ResUnNet, self).__init__()
        resnet_img = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-34
        modules_img = list(resnet_img.children())[:-2]
        self.resnet_img = nn.Sequential(*modules_img)

        # self.dropout = nn.Dropout(0.5)
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
        # result = self.affine_classifier(self.dropout(avg_feature))
        result = self.affine_classifier(avg_feature)
        return result

if __name__ == '__main__':
    net = ResUnNet(num_classes=1)
    images = torch.rand(2, 3, 512, 512)
    output = net(images)
    print(output.shape)
