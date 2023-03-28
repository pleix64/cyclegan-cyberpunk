#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 02:11:11 2022

@author: pleiades486
"""

import itertools
import os
import time

import torch
import torch.nn as nn
import torchvision.utils as vutils

import utils

from cyclegan import Generator as cycG
from cyclegan import Discriminator as cycD

def _weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    #FIXME: InstanceNorm is recommended for cyclegan, 
    # but it seems not working for data.weight initialization
    # Does BatchNorm with batch_size=1 works as the same as 
    # using InstanceNorm? 
    # also look up what the model initialized by default
    #elif classname.find('InstanceNorm') != -1:
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
class Model(object):
    def __init__(self, name, device, data_loader, test_data_loader, 
                 channels, img_size, num_blocks):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.channels = channels
        self.img_size = img_size
        self.num_blocks = num_blocks
        assert self.name == 'cyclegan'
        
        self.netG_AB = cycG(self.channels, self.num_blocks)
        self.netG_AB.apply(_weight_init)
        self.netG_AB.to(self.device)
        
        self.netG_BA = cycG(self.channels, self.num_blocks)
        self.netG_BA.apply(_weight_init)
        self.netG_BA.to(self.device)
        
        self.netD_A = cycD(self.channels)
        self.netD_A.apply(_weight_init)
        self.netD_A.to(self.device)
        
        self.netD_B = cycD(self.channels)
        self.netD_B.apply(_weight_init)
        self.netD_B.to(self.device)
        
        self.optim_G = None
        self.optim_D_A = None
        self.optim_D_B = None
        self.loss_adv = nn.MSELoss()
        self.loss_cyc = nn.L1Loss()
        self.loss_iden = nn.L1Loss()
        
    @property 
    def generator_AB(self):
        return self.netG_AB
        
    @property 
    def generator_BA(self):
        return self.netG_BA
        
    @property 
    def discriminator_A(self):
        return self.netD_A
        
    @property 
    def discriminator_B(self):
        return self.netD_B
        
    def create_optim(self, lr, alpha=0.5, beta=0.999):
        self.optim_G = torch.optim.Adam(itertools
                                        .chain(self.netG_AB.parameters(),
                                               self.netG_BA.parameters()),
                                        lr=lr,
                                        betas=(alpha, beta))
        self.optim_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                          lr=lr,
                                          betas=(alpha, beta))
        self.optim_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                          lr=lr, 
                                          betas=(alpha, beta))
            
    def train(self, epochs, start_epoch=0, 
              log_interval=100, out_dir='', verbose=True):
        self.netG_AB.train()
        self.netG_BA.train()
        self.netD_A.train()
        self.netD_B.train()
        lambda_cyc = 10
        lambda_iden = 5
        real_label = torch.ones((self.data_loader.batch_size, 1, 30, 30),
                                device=self.device)
        fake_label = torch.zeros((self.data_loader.batch_size, 1, 30, 30),
                                 device=self.device)
        image_buffer_A = utils.ImageBuffer()
        image_buffer_B = utils.ImageBuffer()
        total_time = time.time()
        for epoch in range(start_epoch, epochs):
            batch_time = time.time()
            for batch_idx, data in enumerate(self.data_loader):
                real_A = data['trainA'].to(self.device)
                real_B = data['trainB'].to(self.device)
                    
                # Train netG
                self.optim_G.zero_grad()
                # adversarial loss
                fake_B = self.netG_AB(real_A)
                _loss_adv_AB = self.loss_adv(self.netD_B(fake_B), real_label)
                fake_A = self.netG_BA(real_B)
                _loss_adv_BA = self.loss_adv(self.netD_A(fake_A), real_label)
                adv_loss = (_loss_adv_AB + _loss_adv_BA) / 2
                # cycle loss
                recov_A = self.netG_BA(fake_B)
                _loss_cyc_A = self.loss_cyc(recov_A, real_A)
                recov_B = self.netG_AB(fake_A)
                _loss_cyc_B = self.loss_cyc(recov_B, real_B)
                cycle_loss = (_loss_cyc_A + _loss_cyc_B) / 2
                # identity loss
                _loss_iden_A = self.loss_iden(self.netG_BA(real_A), real_A)
                _loss_iden_B = self.loss_iden(self.netG_AB(real_B), real_B)
                iden_loss = (_loss_iden_A + _loss_iden_B) / 2
                g_loss = adv_loss + lambda_cyc * cycle_loss + lambda_iden * iden_loss
                g_loss.backward()
                self.optim_G.step()
                    
                # Train netD_A
                self.optim_D_A.zero_grad()
                _loss_real = self.loss_adv(self.netD_A(real_A), real_label)
                fake_A = image_buffer_A.update(fake_A)
                _loss_fake = self.loss_adv(self.netD_A(fake_A.detach()), 
                                           fake_label)
                d_loss_A = (_loss_real + _loss_fake) / 2
                d_loss_A.backward()
                self.optim_D_A.step()
                    
                # Train netD_B
                self.optim_D_B.zero_grad()
                _loss_real = self.loss_adv(self.netD_B(real_B), real_label)
                fake_B = image_buffer_B.update(fake_B)
                _loss_fake = self.loss_adv(self.netD_B(fake_B.detach()),
                                           fake_label)
                d_loss_B = (_loss_real + _loss_fake) / 2
                d_loss_B.backward()
                self.optim_D_B.step()
                # for logging only
                d_loss = (d_loss_A + d_loss_B) / 2
                
                if verbose and batch_idx % log_interval == 0 and batch_idx > 0:
                    print('Epoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f} time: {:.2f}'.format(
                        epoch, batch_idx, len(self.data_loader),
                        d_loss.mean().item(), 
                        g_loss.mean().item(),
                        time.time() - batch_time))
                    with torch.no_grad():
                        imgs = next(iter(self.test_data_loader))
                        _real_A = imgs['testA'].to(self.device)
                        _fake_B = self.netG_AB(_real_A)
                        _real_B = imgs['testB'].to(self.device)
                        _fake_A = self.netG_BA(_real_B)
                        viz_sample = torch.cat((_real_A, _fake_B, 
                                                _real_B, _fake_A), 0)
                        vutils.save_image(viz_sample, 
                                          os.path.join(out_dir, 
                                                       'samples_{}_{}.png'
                                                       .format(epoch, batch_idx)),
                                          nrow=self.test_data_loader.batch_size,
                                          normalize=True)
                    batch_time = time.time()
                        
            self.save_to(path=out_dir, name=self.name, verbose=False)
            
        if verbose:
            print('Total train time: {:.2f}'.format(
                time.time() - total_time))
                
    def eval(self, batch_size=None):
        self.netG_AB.eval()
        self.netG_BA.eval()
        self.netD_A.eval()
        self.netD_B.eval()
        if batch_size is None:
            batch_size = self.test_data_loader.batch_size
                
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_data_loader):
                _real_A = data['testA'].to(self.device)
                _fake_B = self.netG_AB(_real_A)
                _real_B = data['testB'].to(self.device)
                _fake_A = self.netG_BA(_real_B)
                viz_sample = torch.cat((_real_A, _fake_B, 
                                        _real_B, _fake_A), 0)
                vutils.save_image(viz_sample, 
                                  'img_{}.png'.format(batch_idx),
                                  nrow=batch_size,
                                  normalize=True)
                    
    def save_to(self, path='', name=None, verbose=True):
        if name is None:
            name = self.name
        if verbose:
            print('\nSaving models to {}_G_AB.pt, etc.'.format(name))
        torch.save(self.netG_AB.state_dict(), 
                   os.path.join(path, '{}_G_AB.pt'.format(name)))
        torch.save(self.netG_BA.state_dict(), 
                   os.path.join(path, '{}_G_BA.pt'.format(name)))
        torch.save(self.netD_A.state_dict(), 
                   os.path.join(path, '{}_D_A.pt'.format(name)))
        torch.save(self.netD_B.state_dict(), 
                   os.path.join(path, '{}_D_B.pt'.format(name)))
        
    def load_from(self, path='', name=None, verbose=True):
        if name is None:
            name = self.name
        if verbose:
            print('\nLoading models from {}_G_AB.pt, etc.'.format(name))
            
        ckpt_G_AB = torch.load(os.path.join(path, '{}_G_AB.pt'.format(name)))
        if isinstance(ckpt_G_AB, dict) and 'state_dict' in ckpt_G_AB:
            self.netG_AB.load_state_dict(ckpt_G_AB['state_dict'], strict=True)
        else:
            self.netG_AB.load_state_dict(ckpt_G_AB, strict=True)
                
        ckpt_G_BA = torch.load(os.path.join(path, '{}_G_BA.pt'.format(name)))
        if isinstance(ckpt_G_BA, dict) and 'state_dict' in ckpt_G_BA:
            self.netG_BA.load_state_dict(ckpt_G_BA['state_dict'], strict=True)
        else:
            self.netG_BA.load_state_dict(ckpt_G_BA, strict=True)
            
        ckpt_D_A = torch.load(os.path.join(path, '{}_D_A.pt'.format(name)))
        if isinstance(ckpt_D_A, dict) and 'state_dict' in ckpt_D_A:
            self.netD_A.load_state_dict(ckpt_D_A['state_dict'], strict=True)
        else:
            self.netD_A.load_state_dict(ckpt_D_A, strict=True)
            
        ckpt_D_B = torch.load(os.path.join(path, '{}_D_B.pt'.format(name)))
        if isinstance(ckpt_D_B, dict) and 'state_dict' in ckpt_D_B:
            self.netD_B.load_state_dict(ckpt_D_B['state_dict'], strict=True)
        else:
            self.netD_B.load_state_dict(ckpt_D_B, strict=True)

