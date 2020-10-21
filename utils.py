import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import models
import cv2
import torch

# vgg = models.vgg16(pretrained=True)
#
# img_path = './datasets/images/0003.jpg'
# img = cv2.imread(img_path)
# timg = torch.tensor(img,dtype=torch.float32)
# timg = torch.unsqueeze(timg,0).permute(0,3,1,2)
# print(timg.shape)
class LayerActivitions():
    features = None
    def __init__(self,model,layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self,module,input,output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()

# 将vgg 特征标记，后续vgg在调用时会被监控，也就是使用hook入vgg
# conv_out = LayerActivitions(vgg.features,30)
# o = vgg(Variable(timg))
# conv_out.remove()
# act = conv_out.features
# print(vgg)
# fig = plt.figure(figsize=(40,20))
# fig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
# for i in range(30):
#     ax = fig.add_subplot(6, 9, i+1, xticks=[], yticks=[])
#     ax.set_title(i)
#     print(act[0][i])
#     ax.imshow(act[0][i].data.numpy())
# plt.show()