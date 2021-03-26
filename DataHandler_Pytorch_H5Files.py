# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:55:41 2021

@author: Lisbeth Clausen and Thea Rønn
"""

######################################## Datahandler for loading h5 files###################################################
import torch
import numpy as np
import h5py
import torchvision
import os
from os.path import isfile, join
from torchvision import transforms
import random
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time

#input
    #root: the root af the datafiles
    #channels: Number of channels for the loaded images
    #subset: 'train', 'val' or 'test'
    #slices: One can either choose 'All' or [start,stop]
    # machine: 'Windows' or 'Cluster': This sets the datapaths
#NOTE:
    #It is assumes in getitem, that you either load train and target or just test.

class dataHandler(torchvision.datasets.VisionDataset):
   
    def __init__(self, root, channels = 8, subset='train', slices='All', machine='Windows'):
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.data_path = []
        self.target_path = []
        self.channels = channels
        self.slices = slices
        self.maxx = 7.6665417390625 # Given max value
        self.length_datapath=0
        self.length = 0
        self.machine=machine
        #self.FullImageData=0
        #self.FullTargetData=0
        start_time = time.time()  
        
#-----------loadDatapath: Loads the directions for the subfolders-------------------        
        def loadSubpath(file_path):
            images_dir=[os.path.join(file_path, o) for o in os.listdir(file_path) 
                        if os.path.isdir(os.path.join(file_path,o))]
            return images_dir
        
#-----------loadDatapath: Loads the directions for the files-------------------                
        def loadDatapath(file_path):
            images_dir = [join(file_path, f) for f in os.listdir(file_path) if isfile(join(file_path, f))]
            return images_dir
        
#-----------scale_image: Scale images-------------------------------         
        def scale_image(data, self):
            #data format: images, pixelx,pixely,ch
            data = np.swapaxes(data,1,3)
            #data format: images, ch,pixely,pixelx
            data = data[:,:,2:98,2:98]
            data= np.clip(data,0,1)
            data=data/self.maxx
            return data

#-----------ReadH5FIle: Given a path, reads the file and saves as numpy array---      
        def ReadH5File(file_path, self):
            File = h5py.File(file_path,'r')
            #only include this i we want selected channels
            selected_chs = list(np.linspace(start=0, stop=31, num=self.channels).astype(int))
            data = np.array(File['data']['value'][:,:,:,selected_chs])
            # if we want all channels
            #data = np.array(File['data']['value'][:,:,:,:])
            return data
            
        
#---------- sets the paths for collecting the data-----------------------------        
        if self.machine=='Windows':
            trainpath='\\train_images'
            valpath='\\val_images'
            testpath='\\test_images'
            targetpath_train='\\train_target'
            targetpath_val='\\val_target'
            targetpath_test='\\test_target'
            #trainpath = '\OldData\ART_train_images'
            #valpath = '\OldData\ART_val_images'
            #testpath = '\OldData\ART_test_images'
            #targetpath_train='\OldData\\train_target'
            #targetpath_val='\OldData\\val_target'
            #targetpath_test='\OldData\\test_target'
            


    #Missing correct paths
        if self.machine=='Cluster':
            trainpath='/projections/train_data'
            valpath='/projections/val_data'
            testpath='/projections/test_data'
            targetpath_train='/projections/target_data/train'
            targetpath_val='/projections/target_data/val'
            targetpath_test='/projections/target_data/test'
            


        # makes a list of all subdirections in the projection folder
        if self.subset == "train":
            self.data_path = loadDatapath(self.root + trainpath)
            print('Training files:' , self.data_path)
            self.target_path = loadDatapath(self.root + targetpath_train)
            self.length_datapath=len(self.data_path)
 
        elif self.subset == "val":
            self.data_path = loadDatapath(self.root + valpath)
            print('Validation files:' , self.data_path)
            self.target_path = loadDatapath(self.root + targetpath_val)
            self.length_datapath=len(self.data_path)
            
        elif self.subset == "test":
            self.data_path = loadDatapath(self.root + testpath)
            print('Test files:' , self.data_path)
            self.target_path = loadDatapath(self.root + targetpath_test)
            self.length_datapath=len(self.data_path)
        
        
        # loop over every subfolder
        
        for i in range(self.length_datapath):
            ImageData=ReadH5File(self.data_path[i], self) 
            Targets=ReadH5File(self.target_path[i], self)
            
            #append
            if i==0:
                self.FullImageData=ImageData
                self.FullTargetData=Targets

            else:
                self.FullImageData=np.concatenate((self.FullImageData,ImageData),axis=0)
                self.FullTargetData = np.concatenate((self.FullTargetData, Targets ), axis=0) 
                
        # choose which slices of the data to work with        
        if self.slices != 'All':
            print('Slices chosen: ', slices[0],slices[1])
            self.FullImageData = self.FullImageData[self.slices[0]:self.slices[1],:,:]
            self.FullTargetData = self.FullTargetData[self.slices[0]:self.slices[1],:,:]      
                
        
        # Scale images
        self.length=self.FullImageData.shape[0]
        self.FullImageData=scale_image(self.FullImageData, self)
        self.FullTargetData=scale_image(self.FullTargetData, self)
        
        # transform from numpy to tensor
        self.FullImageData=torch.from_numpy(self.FullImageData)
        self.FullTargetData=torch.from_numpy(self.FullTargetData) 
        print('The shape of the full dataset is', self.FullImageData.shape)
        print('The shape of the full target set is', self.FullTargetData.shape) 
        print(" the length is", self.length)
                         
        print("--- %s seconds ---" % (time.time() - start_time))            
            
          
#--------------------------Accessor function----------------------------------
#---------Getitem: Returns the images of the given index----------------------            
    def __getitem__(self,index):
            data_img=self.FullImageData[index]
            target_img=self.FullTargetData[index] 
            return data_img, target_img
            
#---------les: Returns the number of files loaded in a folder
    def __len__(self):
        return self.length

