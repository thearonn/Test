# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:28:28 2020

@author: IX
"""

# missing relevant comments
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
#from argparse import ArgumentParser
from dataStructures import DataSet
from dataStructures import CTData
from ct_data_generator import CTDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time

#input
    #root: the root af the datafiles
    #transform: two options None or 'YES'
    #channels: Number of channels for the loaded images
    #Loadtrain: Do we wish to load training data - two options 'Y' or 'N'
    #Loadtarget: Do we wish to load target data - two options 'Y' or 'N'
    #Loadtest: Do we wish to load test data - two options 'Y' or 'N'
#WARNING:
    #It is assumes in getitem, that you either load train and target or just test.
    # this should be changes, but for now it will do

#root=root = os.path.expanduser('\\Users\IX\Desktop\Spectral_CT_Bachelorproject\data_ct')
#div=np.array([2/6, 1/6, 1/6, 1/6, 1/6])   
#Projections=np.array([9,18,20,24,26])
class dataHandler(torchvision.datasets.VisionDataset):
   
    def __init__(self, root, channels = 8, div=[], proj=[], subset='train', slices='All', machine='Windows'):
        self.root = os.path.expanduser(root)
        #self.transform = transform
        self.div=div
        self.proj=proj
        self.subset = subset
        self.data_path = []
        self.target_path = []
        self.channels = channels
        self.slices = slices
        #self.trans=[]
        self.maxx = 7.6665417390625 # Given max value
        self.length_datapath=0
        self.length = 0
        self.machine=machine
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
#-----------scale_train: Scale the training data-------------------------------         
        def scale_projim(data, self):
            data = np.swapaxes(data,0,2)
            data = np.swapaxes(data,1,3)
            data=np.swapaxes(data,2,3)
            data = data[:,:,2:98,2:98]
            data= np.clip(data,0,1)
            data=data/self.maxx
            return data
   
        def scale_image(data, self):
            #images, pixelx,pixely,ch
            data = np.swapaxes(data,1,3)
            #images, ch,pixely,pixelx
            data = data[:,:,2:98,2:98]
            data= np.clip(data,0,1)
            data=data/self.maxx
            #data=data*10
            return data
#-------------Generate data--------------------------------------------
        def GenerateData(DataPath, slice_id,slice_no, proj_no,sliceDim ):
            ch_no = channels
            channels_used = list(np.linspace(start=0, stop=80, num=ch_no).astype(int))
            image_size = [100,100] 
            object_scale = 1
            data_set =  DataSet.DataSet(DataPath)
            slice_id=slice_id
            slice_no=slice_no
            if slice_no>sliceDim:
                slice_no=sliceDim-slice_id
            
            data_set.sinogram_data.setLoadSliceZ(slice_id,slice_id+slice_no)
            data_set.sinogram_data.loadData()
            proj_no=proj_no
            print('The used projection nuber is', proj_no)
            data_set.sinogram_data.reduceProjNo(proj_no, data_set.geostruct)
            data_set.sinogram_data.selectChannels(channels_used)
            sim_conf = {'geo_type'       : 'fan', \
                        'proj_no' : proj_no
                            }    
        
            data_gen = CTDataGenerator(sim_conf) # initilizae and generate data
            vectors = data_gen.setGeometry(data_set.geostruct)
            data_gen.generateFromProjections(data_set.sinogram_data.data, image_size, object_scale)
            FBP_data = CTData.ReconstructionData(DataPath)
            FBP_data.data,  interpol_sinograms = data_gen.get_data()
            FBP_data.data = 10*FBP_data.data #hvorfor ganger vi med 10
            return FBP_data.data

#-----------ReadH5FIle: Given a path, reads the file and saves as numpy array---      
        def ReadH5File(file_path, self):
            File = h5py.File(file_path,'r')
            selected_chs = list(np.linspace(start=0, stop=80, num=self.channels).astype(int))
            data = np.array(File['data']['value'][:,:,:,selected_chs])
            #data = data.clip(min=0)
            return data
            
        
#-----------------------------------------------------------------------------        
        if self.machine=='Windows':
            trainpath='\projections\\train_data'
            valpath='\projections\\val_data'
            testpath='\projections\\test_data'
            targetpath_train='\projections\\target_data\\train'
            targetpath_val='\projections\\target_data\\val'
            targetpath_test='\projections\\target_data\\test'

        if self.machine=='Cluster':
            trainpath='/Sinograms/Train_data'
            valpath='/Sinograms/Val_data'
            testpath='/Sinograms/Test_data'
            targetpath_train='/Sinograms/Target_data/Train'
            targetpath_val='/Sinograms/Target_data/Val'
            targetpath_test='/Sinograms/Target_data/Test'
            


        # makes a list of all subdirections in the projection folder
        if self.subset == "train":
            self.data_path = loadSubpath(self.root + trainpath)
            self.target_path = loadDatapath(self.root + targetpath_train)
            self.length_datapath=len(self.data_path)
 
        elif self.subset == "val":
            self.data_path = loadSubpath(self.root + valpath)
            self.target_path = loadDatapath(self.root + targetpath_val)
            self.length_datapath=len(self.data_path)
            
        elif self.subset == "test":
            self.data_path = loadSubpath(self.root + testpath)
            self.target_path = loadDatapath(self.root + targetpath_test)
            self.length_datapath=len(self.data_path)
        
        #defining parameters
        ch_no = channels
        channels_used = list(np.linspace(start=0, stop=80, num=ch_no).astype(int))
        image_size = [100,100] 
        object_scale = 1 # This is a scaling factor for reconsucted objects, keep at 1
        Projections=self.proj
        
        # loop over every subfolæder
        for i in range(self.length_datapath):
            # Create dataset
            data_set =  DataSet.DataSet(self.data_path[i])
            data_set.sinogram_data.loadData()
            # find the dimensions of the dataset
            DIM=data_set.sinogram_data.DIM()
            #Dim[2] is now the number of slices this ample contains
            #now we define the batches we want the data split into
            part=np.ones(len(self.div), dtype=int)*math.ceil((self.div[0])*DIM[2])
            #part=np.array([math.ceil((self.div[0])*DIM[2]),math.ceil((self.div[1])*DIM[2]), math.ceil((self.div[2])*DIM[2]), 
            #math.ceil((self.div[3])*DIM[2]), math.ceil((self.div[4])*DIM[2])])
            #randomize which sequence of slices that gets which projections
            indices = np.arange(part.shape[0])
            np.random.shuffle(indices)
            part = part[indices]
            Projections = Projections[indices]
            print('*******The part array is', part)
            print('************The projections are', Projections)

            if self.subset=="test":
                ImageData=GenerateData(self.data_path[i], 0, DIM[2], Projections[0], DIM[2])
            
            if self.subset=="val" :
                UsedSlices=0
                for j in range(len(self.div)):
                    Data=GenerateData(self.data_path[i], UsedSlices, part[j], Projections[j], DIM[2])
                    UsedSlices+=part[j]
                    # append data
                    if j==0:
                        ImageData=Data
                    else:
                        ImageData = np.concatenate((ImageData,Data),axis=2) 

           # create images with different projections
            if self.subset=="train":
                UsedSlices=0
                for j in range(len(self.div)):
                    Data=GenerateData(self.data_path[i], UsedSlices, part[j], Projections[j], DIM[2])
                    UsedSlices+=part[j]
                    # append data
                    if j==0:
                        ImageData=Data
                    else:
                        ImageData = np.concatenate((ImageData,Data),axis=2)
            #append
            if i==0:
                self.FullImageData=ImageData
                self.FullTargetData=ReadH5File(self.target_path[i], self) 
                #if self.subset=='test':
                    #self.FullImageData=self.FullImageData[:-10]
                  # self.FullTargetData=self.FullTargetData[:-10]
                
                                
            else:
                self.FullImageData=np.concatenate((self.FullImageData,ImageData),axis=2)
                self.FullTargetData = np.concatenate((self.FullTargetData, ReadH5File(self.target_path[i], self) ), axis=0) 
               # if self.subset=='test':
                   # self.FullImageData=self.FullImageData[:-10]
                    #self.FullTargetData=self.FullTargetData[:-10]
                
                
                
        if self.slices != 'All':
            print('slices: ', slices[0],slices[1])
            self.FullImageData = self.FullImageData[self.slices[0]:self.slices[1],:,:]
            self.FullTargetData = self.FullTargetData[self.slices[0]:self.slices[1],:,:] if not self.subset == 'test' else None      
                
        self.length=self.FullImageData.shape[2]
        self.FullImageData=scale_projim(self.FullImageData, self)
        self.FullTargetData=scale_image(self.FullTargetData, self)
        
        self.FullImageData=torch.from_numpy(self.FullImageData)
        self.FullTargetData=torch.from_numpy(self.FullTargetData) 
        print('The shape of the full dataset is', self.FullImageData.shape)
        print('The shape of the full target set is', self.FullTargetData.shape) 
                         
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

