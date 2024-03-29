from models.resnet_block import BasicBlock,Bottleneck
import torch
import torch.nn as nn
import yaml
import logging
import os
import cv2
import numpy as np
from torch.optim import Adam

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=np.inf)

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self._check_branches(
            num_branches, num_blocks, num_inchannels, num_channels
        )
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) != NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) != NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) != NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or \
            self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM)
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches =self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []

        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j],num_inchannels[i],1,1,0,bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode = 'nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)

                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i-j-1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j],num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    # x也是一个列表，内部包含各层要输入的初始图像
    def forward(self,x):
        if self.num_branches == 1:
            return  [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            # print(x[0].shape)
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1,self.num_branches):
                if i == j:
                    y = y + x[i]
                else:
                    # print('x[j]',x[j].shape)
                    # print('x_f_s',self.fuse_layers[i][j](x[j]).shape)
                    # print('x_f',self.fuse_layers[i][j])
                    # print('y',y.shape)
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append((self.relu(y)))

        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class HighResolutionNet(nn.Module):
    def __init__(self,config):
        super(HighResolutionNet, self).__init__()
        self.inplanes = 64
        extra = config['MODEL']['EXTRA']
        super(HighResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(32,momentum=BN_MOMENTUM)
        self.bn4 = nn.BatchNorm2d(1, momentum=BN_MOMENTUM)
        # _make_layer(基本残差块，入通道，出通道，残差块数目)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256],num_channels)
        self.stage2,pre_stage_channels = self._make_stage(self.stage2_cfg,num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels,num_channels
        )
        self.stage3,pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        #  final_inp_channels = sum(pre_stage_channels)
        self.final_layers = self._make_final_layers(config,pre_stage_channels[0])
        print(self.final_layers)
        # 反卷积使得分辨率保持与最上方一致
        self.deconv_layers = self._make_deconv_layers(config, pre_stage_channels[0])
        self.num_deconvs = extra['DECONV']['NUM_DECONVS']
        self.deconv_config = config['MODEL']['EXTRA']['DECONV']
        self.loss_config = None
        self.pretrained_layers = config['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        self.conv2_1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.conv4_1 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)
        self.conv1_fin = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv_sum_c = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

    def _make_final_layers(self, config, input_channels):
        dim_tag = config['MODEL']['NUM_JOINTS'] if config['MODEL']['NUM_JOINTS'] else 1
        extra = config['MODEL']['EXTRA']

        final_layers = []
        output_channels = config['MODEL']['NUM_JOINTS'] + dim_tag \
            if config['LOSS']['WITH_AE_LOSS'][0] else config['MODEL']['NUM_JOINTS']
        final_layers.append(
            nn.Conv2d(
                in_channels = input_channels,
                out_channels= output_channels,
                kernel_size= extra['FINAL_CONV_KERNEL'],
                stride = 1,
                padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
            )
        )

        # TODO
        deconv_cfg = extra['DECONV']
        for i in range(deconv_cfg['NUM_DECONVS']):
            input_channels = deconv_cfg['NUM_CHANNELS'][i]
            output_channels = config['MODEL']['NUM_JOINTS'] + dim_tag \
                if config['LOSS']['WITH_AE_LOSS'][i+1] else config['MODEL']['NUM_JOINTS']
            final_layers.append(nn.Conv2d(
                in_channels = input_channels,
                out_channels = output_channels,
                kernel_size = extra['FINAL_CONV_KERNEL'],
                stride=1,
                padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
            ))

        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self,config,input_channels):
        dim_tag = config['MODEL']['NUM_JOINTS'] if config['MODEL']['TAG_PER_JOINT'] else 1
        extra = config['MODEL']['EXTRA']
        deconv_cfg = extra['DECONV']

        deconv_layers = []
        for i in range(deconv_cfg['NUM_DECONVS']):
            if deconv_cfg['CAT_OUTPUT'][i]:
                final_output_channels = config['MODEL']['NUM_JOINTS'] + dim_tag if config['LOSS']['WITH_AE_LOSS'][i] else config['MODEL']['NUM_JOINTS']
                input_channels += final_output_channels

            output_channels = deconv_cfg['NUM_CHANNELS'][i]
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(deconv_cfg['KERNEL_SIZE'][i])

            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels = input_channels,
                    out_channels = output_channels,
                    kernel_size = deconv_kernel,
                    stride = 2,
                    padding = padding,
                    output_padding = output_padding,
                    bias = False
                ),
                nn.BatchNorm2d(output_channels, momentum = BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
            for _i in range(config['MODEL']['EXTRA']['DECONV']['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels)
                ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j==i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                    transition_layers.append(nn.Sequential(*conv3x3s))

                return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules -1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self,x):
        x =self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # print(x.shape)
        # x = self.bn2(x)
        # x = self.relu(x)

        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        fin_branch2 = self.conv2_1(y_list[1])
        fin_branch2 = nn.functional.interpolate(fin_branch2,scale_factor=2,mode='nearest')
        fin_branch3 = self.conv3_1(y_list[2])
        fin_branch3 = nn.functional.interpolate(fin_branch3, scale_factor=4, mode='nearest')
        fin_branch4 = self.conv4_1(y_list[3])
        fin_branch4 = nn.functional.interpolate(fin_branch4, scale_factor=8, mode='nearest')

        result = self.conv_sum_c(torch.add(torch.add(torch.add(y_list[0],fin_branch2),fin_branch3),fin_branch4))
        result = self.bn3(result)
        # fin_output = self.conv1_fin(result)
        # print(fin_output)
        # print('fin_output',fin_output.shape)
        # fin_output = nn.functional.interpolate(result,scale_factor=4,mode='nearest')
        fin_output = self.conv1_fin(result)
        return fin_output

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    path = '../config/module.yaml'
    config = yaml.load(open(path,'r'),yaml.SafeLoader)
    print(config)
    net = HighResolutionNet(config)
    img_path = '../datasets/images/0003.jpg'
    img = cv2.imread(img_path)

    img_tensor = torch.tensor(img,dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0).permute(0,3,1,2)
    sal_img_tensor = net(img_tensor)
    print('sal_image',sal_img_tensor.shape)
    sal_img = np.squeeze(torch.sigmoid(sal_img_tensor).data.numpy())
    print(sal_img.shape)
    print(sal_img * 255)
    # print(img_tensor.shape) # torch.Size([1, 3, 256, 256])
    cv2.imshow('images.jpg',sal_img * 255)

    cv2.waitKey(0)
    # print(net)