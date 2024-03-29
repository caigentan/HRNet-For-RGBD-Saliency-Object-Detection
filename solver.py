from models.RGBHRnet import HighResolutionNet,weights_init
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from datasets.dataset import ImageDataTrain,ImageDataTest,get_loader
from torch.utils import data
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import os
import time
import yaml
class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config['iter_size']
        self.show_everyone = config['show_everyone']
        self.lr_decay_epoch = [15,]
        self.build_model()
        if config['mode'] == 'test':
            print('Loading pre-trained model from %s...' % self.config['model'])
            if self.config['cuda']:
                self.net.load_state_dict(torch.load(self.config['model']))
            else:
                self.net.load_state_dict(torch.load(self.config['model'], map_location='cpu'))
            self.net.eval()

        # print the network information and parameter numbers

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

        self.lr = self.config['lr']
        self.wd = self.config['wd']

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                              weight_decay=self.wd)
        self.print_network(self.net, 'RGBHRNet Structure')

    def test(self):
        mode_name = 'tyc'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = Variable(images)
                if self.config['cuda']:
                    images = images.cuda()
                preds = self.net(images)
                pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                multi_fuse = 255 * pred
                print(im_size)
                multi_fuse = cv2.resize(multi_fuse,(im_size[1],im_size[0]))
                print(name)
                cv2.imwrite(os.path.join(self.config['test_fold'], name[:-4] + '_' + mode_name + '.png'), multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config['batch_size']
        aveGrad = 0
        for epoch in range(self.config['epoch']):
            r_sal_loss = 0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue

                sal_image, sal_label = Variable(sal_image), Variable(sal_label)
                if self.config['cuda']:
                    # cudnn.benchmark = True
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

                sal_pred = self.net(sal_image)
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                sal_loss = sal_loss_fuse / (self.iter_size * self.config['batch_size'])
                r_sal_loss += sal_loss.data

                sal_loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.show_everyone // self.config['batch_size']) == 0:
                    if i == 0:
                        x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                        epoch, self.config['epoch'], i, iter_num, r_sal_loss / x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_sal_loss = 0

            if (epoch + 1) % self.config['epoch_save'] == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config['save_folder'], epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                                      weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config['save_folder'])

