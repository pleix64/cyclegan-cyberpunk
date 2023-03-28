#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:41:47 2022

@author: pleiades486
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils
from datasets import ImageDataset

from trainer import Model

FLAGS = None

def main():
    device = torch.device('cuda:0' if FLAGS.cuda else 'cpu')
    
    if FLAGS.train:
        print('Loading data...\n')
        transform = [transforms.Resize(int(FLAGS.img_size * 1.12), 
                                       transforms.InterpolationMode.BICUBIC),
                     transforms.RandomCrop((FLAGS.img_size, FLAGS.img_size)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        train_data = ImageDataset(os.path.join(FLAGS.data_dir, FLAGS.dataset),
                                  transform=transform, 
                                  unaligned=True, 
                                  mode='train')
        dataloader = DataLoader(train_data, batch_size=FLAGS.batch_size, 
                                shuffle=True, num_workers=2)
        test_data = ImageDataset(os.path.join(FLAGS.data_dir, FLAGS.dataset),
                                 transform=transform, 
                                 unaligned=True,
                                 mode='test')
        test_dataloader = DataLoader(test_data, batch_size=FLAGS.test_batch_size, 
                                     shuffle=True, num_workers=2)
        
        print('Creating model...\n')
        model = Model(FLAGS.model, device, dataloader, test_dataloader, 
                      FLAGS.channels, FLAGS.img_size, FLAGS.num_blocks)
        model.create_optim(FLAGS.lr)
        if FLAGS.start_epoch != 0:
            model.load_from(FLAGS.checkpoint_dir)
        
        # Train
        model.train(FLAGS.epochs, FLAGS.start_epoch, FLAGS.log_interval, FLAGS.out_dir, True)
        model.save_to()
    else:
        # Caution: test_dataloader may not be available in this scope. FIXME!
        # test first!
        model = Model(FLAGS.model, device, None, test_dataloader, 
                      FLAGS.channels, FLAGS.img_size, FLAGS.num_blocks)
        model.load_from(FLAGS.out_dir)
        # eval func seems not defined in this way. test first! FIXME!
        model.eval(model=1, batch_size=FLAGS.batch_size)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cyclegan', help='Currently only support cyclegan model')
    parser.add_argument('--cuda', action='store_true', help='enable CUDA')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disable CUDA')
    parser.set_defaults(cuda=True)
    parser.add_argument('--train', action='store_true', help='train mode')
    parser.add_argument('--eval', dest='train', action='store_false', help='eval mode')
    parser.set_defaults(train=True)
    parser.add_argument('--start_epoch', type=int, default=0, help='Used for resume training.')
    parser.add_argument('--checkpoint_dir', type=str, default='trained', help='Directory for last trained model checkpoint.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for dataset.')
    parser.add_argument('--dataset', type=str, default='cyberpunk', help='Dataset name.')
    parser.add_argument('--out_dir', type=str, default='output', help='Directory for output.')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='size of batches in training')
    parser.add_argument('--test_batch_size', type=int, default=4, help='size of batches in inference')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='size of images')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--num_blocks', type=int, default=9, help='number of residual blocks')
    parser.add_argument('--log_interval', type=int, default=100, help='interval between logging and image sampling')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()
    
    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        if FLAGS.cuda:
            torch.cuda.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
    
    cudnn.benchmark = True
    
    if FLAGS.train:
        utils.clear_folder(FLAGS.out_dir)
        
    log_file = os.path.join(FLAGS.out_dir, 'log.txt')
    print("Logging to {}\n".format(log_file))
    sys.stdout = utils.StdOut(log_file)

    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA version: {}\n".format(torch.version.cuda))

    print(" " * 9 + "Args" + " " * 9 + "|    " + "Type" + \
          "    |    " + "Value")
    print("-" * 50)
    for arg in vars(FLAGS):
        arg_str = str(arg)
        var_str = str(getattr(FLAGS, arg))
        type_str = str(type(getattr(FLAGS, arg)).__name__)
        print("  " + arg_str + " " * (20-len(arg_str)) + "|" + \
              "  " + type_str + " " * (10-len(type_str)) + "|" + \
              "  " + var_str)
    
    main()
    