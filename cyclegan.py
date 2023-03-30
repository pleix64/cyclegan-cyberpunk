#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 00:10:18 2022

@author: pleiades486
"""

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(channels, channels, 3),
                                   nn.InstanceNorm2d(channels),
                                   nn.ReLU(inplace=True),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(channels, channels, 3),
                                   nn.InstanceNorm2d(channels))
    def forward(self, x):
        return x + self.block(x)
    

class Generator(nn.Module):
    def __init__(self, channels, num_blocks=9):
        super().__init__()
        self.channels = channels
        residual_blocks = [ResidualBlock(256)] * num_blocks
        
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            *self._create_layer(self.channels, 64, 7, stride=1, padding=0),
            # downsampling
            *self._create_layer(64, 128, 3, stride=2, padding=1),
            *self._create_layer(128, 256, 3, stride=2, padding=1),
            # residual blocks
            *residual_blocks, 
            # upsampling
            *self._create_layer(256, 128, 3, stride=2, padding=1, transposed=True),
            *self._create_layer(128, 64, 3, stride=2, padding=1, transposed=True),
            # output 
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
            )
        
    def _create_layer(self, size_in, size_out, kernel_size, stride=2, 
                      padding=1, transposed=False):
        layers = []
        if transposed:
            layers.append(nn.ConvTranspose2d(size_in, size_out, kernel_size,
                                             stride=stride, padding=padding,
                                             output_padding=1))
        else:
            layers.append(nn.Conv2d(size_in, size_out, kernel_size,
                                    stride=stride, padding=padding))
        layers.append(nn.InstanceNorm2d(size_out))
        layers.append(nn.ReLU(inplace=True))
        return layers
    
    def forward(self, x):
        return self.model(x)
    
    
class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
       
        self.model = nn.Sequential(
            *self._create_layer(self.channels, 64, stride=2, normalize=False),
            *self._create_layer(64, 128, stride=2),
            *self._create_layer(128, 256, stride=2),
            *self._create_layer(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
            )
       
    def _create_layer(self, size_in, size_out, stride, normalize=True):
        layers = [nn.Conv2d(size_in, size_out, 4, stride=stride, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(size_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
   
    def forward(self, x):
        return self.model(x)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    