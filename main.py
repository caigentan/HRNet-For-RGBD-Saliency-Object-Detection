from datasets.dataset import ImageDataTrain,ImageDataTest,get_loader
from solver import Solver
import models.RGBHRnet
import torch
import os
import yaml
def get_test_info(sal_mode = 'e'):
    if sal_mode == 'e':
        image_root = './datasets/ECSSD/Imgs/'
        image_source = './datasets/ECSSD/test_list.lst'
    return image_root, image_source
def main(config):
    if config['mode'] == 'train':
        train_loader = get_loader(config)
        run = 0
        while os.path.exists("%s/run-%d" % (config['save_folder'], run)):
            run += 1
        os.mkdir("%s/run-%d" % (config['save_folder'], run))
        os.mkdir("%s/run-%d/models" % (config['save_folder'], run))
        config['save_folder'] = "%s/run-%d" % (config['save_folder'], run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config['mode'] == 'test':
        config['test_root'], config['test_list'] = get_test_info(config['sal_mode'])
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config['test_fold']): os.mkdir(config['test_fold'])
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")

if __name__ == '__main__':
    path = './config/module.yaml'
    config = yaml.load(open(path, 'r'), yaml.SafeLoader)
    main(config)
