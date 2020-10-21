class HighResolutionModule(nn.Module):
    # num_branches:分支数目
    # blocks:
    # num_blocks:分的块数目
    # num_inchannels:输入的通道数目
    # num_channels:输出的通道数目
    # fuse_method:混合方式
    # multi_scale_output:多级融合
    # net = HighResolutionModule(3, Bottleneck, )
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        # 首先检查分支数与num_blocks(每个分支分别所拥有的的块数目)长度，num_inchannels(各个块的输入通道数)长度
        # num_channels(各个块的输出通道数)长度是否一致
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        # 入通道数
        self.num_inchannels = num_inchannels
        # 融合方式
        self.fuse_method = fuse_method
        # 分支数目
        self.num_branches = num_branches
        # 多尺度输出
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        # 层与层之间的融合
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

# 检测分支数目
    # 判断分支数(num_branches,num),块数(num_blocks),len(num_inchannel),len(num_channels)是否一致
    # num_inchannels表示输入通道列表，num_channels表示输出通道列表
    # num_block是一个列表，表示对应index的列表分别拥有几个模块

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 创建一个分支
    # 单个分支内部分辨率相同，一个分支由num_blocks[branch_index]个block组成,block可以是两种ResNet模块中的一种(BasicBlock和Bottleneck)
    # 1、首先判断是否降维或者输入输出的通道(num_inchannels[branch_index]和num_channels[branch_index] * block.expansion(通道扩张数))是否一致，
    # 如果不一致则使用1x1卷积进行维度升/降，后接BN，不使用ReLU
    # 2、顺序搭建num_block[branch_index]个block，第一个block需要考虑是否降维度的情况，所以单独拿出来，后面1~num_block[brnach_index]完全一致，使
    # 用循环搭建就可以，另外在执行完第一个block后num_inchannels[branch_index]重新赋值为num_inchannels[branch_index]*block.expansion
    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    # 创建分支
    # 循环调用make_one_branch函数创建多个分支
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    # 不同层融合，也就是将我们所创建的多个分支融合在一起形成一个整体
    def _make_fuse_layers(self):
        # 如果只有一个分支返回None，说明此时不需要使用融合模块

        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        # 这里猜测 num_inchannels表示不同层所存的通道数
        num_inchannels = self.num_inchannels
        fuse_layers = []
        # 若采用多级输出则 num_branches 反之 1
        # 双层循环作用：
        # 第一层：如果需要产生多分辨率的结果，就双层循环num_branches次,如果只需要产生最高分辨率表示，就将i确定为0.

        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                # 如果 j>i，此时的目标是将所有分支上采样到和i分支相同分辨率并融合，也就是说j所代表的的分支的分辨率比i分支低，
                # 2**(j-i)表示j分支上采样多少倍才能和i分支相同。先用1*1卷积将j分支通道数变得和i分支一致，进而进行BN，然后
                # 依据上采样因子将j分支分辨率上采样到和分支分辨率相同，此处采用的上采样方式是最近邻插值

                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        # scale_factor:放大的尺度比例[1,2,4]，上采样的方式是 'nearest'--最近邻插值
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                # 如果j = i,也就是说自身与自身之间不需要融合 None
                elif j == i:
                    fuse_layer.append(None)
                # 若 j<i ，与j>i时相反，此时j所代表的分辨率i高，此时再次内嵌了一个循环，这个循环的作用是当i-j>1时，也就是说
                # 两个分支之间的分辨率差了不止2倍，此时还是两倍两倍的上采样，例如i-j==2时，j分支的分辨率比i分支大4倍，就需
                # 要上采样2次，循环次数就是2；
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        # 当k=i-j-1时,表示相邻,如当i=2,j=1时，此时仅仅循环一次，并且采用当前模块，此时直接将j分支使用3*3卷积的
                        # 步长为2的卷积下采样(不使用bias)，后接BN，不使用ReLU;

                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        # 当k != i-j-1 时，例如当 i==3,j==1，此时循环两次，先采用当前模块，将j分支使用3*3的步长为2的
                        # 卷积下采样(不使用bias)两倍，后接BN和ReLU,紧跟着在使用 k == i-j-1 的模块，这是为了保证最后一次二倍下
                        # 采样的卷积操作不使用ReLU,也是为了保证融合后特征的多样性
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    # 利用上面的分支，搭建网络
    def forward(self, x):
        # 当仅只有一个分支时，生成该分支，没有融合模块，直接返回
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # 融合模块，当包含不仅一个分支时，先将对应的输入特征输入到对应分支，得到对应分支的输出特征，紧接着执行融合模块
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse