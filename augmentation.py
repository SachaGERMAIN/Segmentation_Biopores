import numpy as np 
import tifffile as tiff
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt
import albumentations as albu
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from glob import glob
import os
from MO.dataset_test import test

IMAGE_PATH = '.\data\raw\galleries\image'
MASK_PATH = '.\data\raw\galleries\mask'




class augmenting_Dataset(Dataset):
    
    def __init__(self, img_path, mask_path):
        self.images = glob(f"{img_path}/*.tif")  
        self.masks = glob(f"{mask_path}/*.tif")
        self.img_path=img_path

    def __len__(self):
        return len(self.images)
       
    def __getitem__(self, idx: int)-> Tensor:
        imgp=self.images[idx]
        maskp=self.masks[idx]
        #Ouverture en np.array
        image=tiff.imread(imgp)
        mask=tiff.imread(maskp)      
        #passage en Tensor
        image=image.astype('int32')
        mask=mask.astype('int32')

        return image, mask,imgp,maskp




def augment_flip_H_V(Dataset): 
    augmentation_HF=albu.HorizontalFlip(p=1)
    augmentation_VF=albu.VerticalFlip(p=1)
    image_dict={"Image":[],"Name_im":[],'Mask':[],'Name_mask':[]}
    #colormaps=ListedColormap(['darkorange','gold','lawngreen']).resampled(256)
    for i in range(len(Dataset)):
        augmented_HF = augmentation_HF(image = Dataset[i][0], mask=Dataset[i][1])
        augmented_VF = augmentation_VF(image = Dataset[i][0], mask=Dataset[i][1])
        image_dict['Image'].append(augmented_HF['image'])
        image_dict['Name_im'].append(Dataset[i][2][-12:-4]+"_HF.tif")
        image_dict['Mask'].append(augmented_HF['mask'])
        image_dict['Name_mask'].append(Dataset[i][3][-7:-4]+"_HF.tif")
        image_dict['Image'].append(augmented_VF['image'])
        image_dict['Name_im'].append(Dataset[i][2][-12:-4]+"_VF.tif")
        image_dict['Mask'].append(augmented_VF['mask'])
        image_dict['Name_mask'].append(Dataset[i][3][-7:-4]+"_VF.tif")

        # fig,axs=plt.subplots(2,2)
        # axs[0,0].pcolormesh(Dataset[i][1], cmap=colormaps, rasterized=True)
        # axs[0,0].set_title('initial mask')
        # axs[0,1].pcolormesh(augmented_HF['mask'], cmap=colormaps, rasterized=True)       ######### Les lignes en commentaires servent à visualiser si les inversions se font bien
        # axs[0,1].set_title('HF Mask')
        # axs[1,0].imshow(Dataset[i][0],cmap='gray',origin='lower')
        # axs[1,0].set_title('Original image')
        # axs[1,1].imshow(augmented_HF['image'],cmap='gray',origin='lower')
        # axs[1,1].set_title('HF image')
        # fig.set_size_inches(10,10)
        # plt.show(block=True)
    return image_dict



xyz=augmenting_Dataset(IMAGE_PATH,MASK_PATH)
aug_list=augment_flip_H_V(xyz)

#### Saving images in a new folder

os.chdir(r"C:\Users\ccaurel\Desktop\Python_Deep_Learning\data\processed\galleries\image")

for i in range(len(aug_list['Image'])):
    tiff.imwrite(aug_list['Name_im'][i],aug_list['Image'][i])

#### Saving masks in a new folder

os.chdir(r'C:\Users\ccaurel\Desktop\Python_Deep_Learning\data\processed\galleries\mask')

for i in range(len(aug_list['Mask'])):
    tiff.imwrite(aug_list['Name_mask'][i],aug_list['Mask'][i])


