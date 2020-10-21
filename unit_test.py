from models.RGBHRnet import HighResolutionNet
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from datasets.dataset import ImageDataTrain,ImageDataTest,get_loader
from torch.utils import data
from torch.optim import Adam
from models.RGBHRnet import weights_init
from torch.autograd import Variable
import torch.nn.functional as F
import os
import yaml

class ImageDataTrain(data.Dataset):
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path

        self.sal_num = 1

    def __getitem__(self, item):

        sal_image = load_img(self.img_path)
        sal_label = load_sal_label(self.label_path)
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sample = {'sal_image':sal_image, 'sal_label':sal_label}
        return sample

    def __len__(self):
        return self.sal_num

def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode ==  'train':
        shuffle = True
        dataset = ImageDataTrain(config['img_path'],config['label_path'])
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=1,
            pin_memory=pin
        )
    return data_loader


def load_img(path):
    img = cv2.imread(path)
    in_ = np.array(img,dtype=np.float32)
    in_ = in_.transpose((2, 0, 1))
    return in_

def load_sal_label(path):
    sal_label = Image.open(path)
    label = np.array(sal_label,dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis,...]
    return label

class Solver(object):
    def __init__(self,train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config['iter_size']
        self.show_every = config['show_every']
        self.lr_decay_epoch = [15, ]
        self.build_model()
        if config['mode'] == 'test':
            print('Loading pre-trained model from %s...' % self.config['model'])
        # if self.config['cuda']:
        #     self.net.load_state_dict(torch.load(self.config['model']))
        # else:
        #     self.net.load_state_dict(torch.load(self.config['model'], map_location='cpu'))
        self.net.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = HighResolutionNet(self.config)
        if self.config['cuda']:
            self.net = self.net.cuda()
        # self.net.train()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)
        # self.net.load_state_dict(torch.load(self.config['load']))

        self.lr = self.config['lr']
        self.wd = self.config['wd']

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)
        self.print_network(self.net, 'RGBHRNet Structure')

    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config['batch_size']
        aveGrad = 0
        for epoch in range(50):
            r_sal_loss = 0
            self.net.zero_grad()
            for i,data_batch in enumerate(self.train_loader):
                sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label = Variable(sal_image), Variable(sal_label)
                if self.config['cuda']:
                    # cudnn.benchmark = True
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()
                sal_pred = self.net(sal_image)
                pred = np.squeeze(torch.sigmoid(sal_pred).cpu().data.numpy())
                multi_fuse = 255 * pred
                # print('multi_fuse', multi_fuse.shape)
                cv2.imwrite(os.path.join('./util_test/', str(epoch) + '_'+ str(i) +'.png'), multi_fuse)

                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                sal_loss = sal_loss_fuse / (self.iter_size * self.config['batch_size'])
                r_sal_loss += sal_loss.data

                sal_loss.backward()

                aveGrad += 1

                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % (self.show_every // self.config['batch_size']) == 0:
                    if i == 0:
                        x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                        epoch, 5000, i, iter_num, r_sal_loss/x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_sal_loss= 0
            # if (epoch + 1) % self.config.epoch_save == 0:
            #     torch.save(self.net.state_dict(), '%s/frames/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            #
            # if epoch in self.lr_decay_epoch:
            #     self.lr = self.lr * 0.1
            #     self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

def main(config):
    if config['mode'] == 'train':
        train_loader = get_loader(config)
        train = Solver(train_loader, None, config)
        train.train()
    else:
        raise IOError("illegal input!!!")

if __name__ == '__main__':
    path = './config/module.yaml'
    config = yaml.load(open(path, 'r'), yaml.SafeLoader)
    main(config)


