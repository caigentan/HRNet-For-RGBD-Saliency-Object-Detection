from utils import LayerActivitions
from models.RGBHRnet import HighResolutionNet
import torch
import cv2
import matplotlib.pyplot as plt

import yaml

path = './config/module.yaml'
config = yaml.load(open(path,'r'),yaml.SafeLoader)

net = HighResolutionNet(config)

img_path = './datasets/images/0003.jpg'
img = cv2.imread(img_path)
timg = torch.tensor(img,dtype=torch.float32)
timg = torch.unsqueeze(timg,0).permute(0,3,1,2)

conv_out = LayerActivitions(net.transition1,0)

out = net(timg)

conv_out.remove()
act = conv_out.features
print(act[0].shape)
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
for i in range(30):
    ax = fig.add_subplot(5, 6, i + 1, xticks=[], yticks=[])
    ax.set_title(i)

    ax.imshow(act[0][i].data.numpy())
plt.show()
