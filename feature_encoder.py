import torch
import torch.nn as nn
import torchvision.models as models

# Load the pre-trained ResNet34 model and fix the paremeters
# resnet = models.resnet34(pretrained=True)
# for param in resnet.parameters():
#     param.requires_grad = False

# Define a feature extractor that outputs the features from the first four layers
class FeatureExtractor(nn.Module):
    def __init__(self, upsample_factor=[2, 4, 8, 8]):
        super(FeatureExtractor, self).__init__()
        self.resnet =  models.resnet34(pretrained=True)
        # self.layer1 = nn.Sequential(
        #     resnet.conv1,
        #     resnet.bn1,
        #     resnet.relu
        # )
        # self.layer2 = nn.Sequential(*list(resnet.layer1.children()))
        # self.layer3 = nn.Sequential(*list(resnet.layer2.children()))
        # self.layer4 = nn.Sequential(*list(resnet.layer3.children()))

        # Add upsampling layers after each convolutional layer
        self.upsample1 = nn.Upsample(scale_factor=upsample_factor[0], mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=upsample_factor[1], mode='bilinear', align_corners=True)
        # self.upsample3 = nn.Upsample(scale_factor=upsample_factor[2], mode='bilinear', align_corners=True)
        # self.upsample4 = nn.Upsample(scale_factor=upsample_factor[3], mode='bilinear', align_corners=True)

    def forward(self, x):
        # x1 = self.layer1(x)
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # x4 = self.layer4(x3)
        # print('the shape of feature',x1.shape,x2.shape,x3.shape,x4.shape) # [n,64,x/2,y/2],[n,64,x/2,y/2],[n,128,x/4,y/4],[n,256,x/8,y/8]
        x1 = self.resnet.conv1(x)
        x1 = self.resnet.bn1(x1)
        x2 = self.resnet.relu(x1)
        x2 = self.resnet.maxpool(x2)
        x2 = self.resnet.layer1(x2)
        # print('the shape of feature',x1.shape,x2.shape) # [n,64,x/2,y/2],[n,64,x/4,y/4]

        # Upsample the features and concatenate them
        x1 = self.upsample1(x1)
        x2 = self.upsample2(x2)
        # x3 = self.upsample3(x3)
        # x4 = self.upsample4(x4)
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.cat((x1[:,:32,:,:], x2[:,:32,:,:]), dim=1)
        return x