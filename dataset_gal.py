##### 
#This is the dataset script, where we define our Dataset objects
#####



#Library imports 
import numpy as np 
import tifffile as tiff
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from glob import glob
import os




class NormalisationDataset(Dataset):                       
   
    def __init__(self, img_directory):   
        self.images = glob(f"{img_directory}/*.tif")  
        self.transform=T.ToTensor()                   

    def __getitem__(self, idx: int) -> Tensor:
        imgp=self.images[idx]
        img = tiff.imread(imgp)
        img=img.astype('int32')
        img=self.transform(img)
        img=img/65536
        return img


    def __len__(self):
        return len(self.images)



def get_normalize_params(data_dir: str) -> tuple[Tensor, Tensor]:
    """
    take normalisation dataset and output mean and std over 3 channels for
    large dataset (using batch processing).
    mean and std are tensor like
    """
    dataset = NormalisationDataset(img_directory=data_dir)
    dataloader = DataLoader(dataset, batch_size=16)  # store data in datloader
    channels_sum, channels_squared_sum, num_batches = (
        0,
        0,
        0
    )  # initialize channel stats

    for data in tqdm(dataloader,unit='batch'):
        # Mean over batch, height and width but no channels
        data_s=data**2
        channels_sum += torch.mean(data.float(),dim=[0,2,3])
        channels_squared_sum += torch.mean(data_s.float(),dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches  # compute mean over batches
    std = (
        channels_squared_sum / num_batches - mean**2
    ) ** 0.5  # compute std over batches

    return mean, std 

class testDataset(Dataset):
    
    def __init__(self, img_path, mask_path):
        self.images = glob(f"{img_path}/*.tif")  
        self.masks = glob(f"{mask_path}/*.tif")
        self.transform=T.ToTensor()
        self.img_path=img_path

    def __len__(self):
        return len(self.images)
       
    def __getitem__(self, idx: int)-> Tensor:
        imgp=self.images[idx]
        maskp=self.masks[idx]
        #Ouverture en np.array (matrice numpy)
        image=tiff.imread(imgp)
        mask=tiff.imread(maskp)   
        #passage en Tensor
        image=image.astype('int32')
        mask=mask.astype('int32')
        imaget=self.transform(image)
        maskt=self.transform(mask)
        #rescale entre 0 et 1
        imaget=imaget/65536
        #normalize
        meani,stdi=get_normalize_params(self.img_path)
        norm_im=T.Normalize(mean=meani,std=stdi)
        image_n=norm_im(imaget)
        #Resizing the image for quicker computation, or possible to add pading ot reach a multiple of 32 for the dimensions (here from 491*491 to 512*512)
        padding=T.Resize(size=(256,256))
        masktr=padding(maskt)
        image_nr=padding(image_n)

        return image_nr, masktr, image, mask
    
class inf_Dataset(Dataset):
    
    def __init__(self, img_path):
        self.images = glob(f"{img_path}/*.tif")  
        self.transform=T.ToTensor()
        self.img_path=img_path

    def __len__(self):
        return len(self.images)
       
    def __getitem__(self, idx: int)-> Tensor:
        imgp=self.images[idx]
        #Ouverture en np.array
        image=tiff.imread(imgp)     
        #passage en Tensor
        image=image.astype('int32')
        imaget=self.transform(image)
        #rescale entre 0 et 1
        imaget=imaget/65536
        #normalize
        meani,stdi=get_normalize_params(self.img_path)
        norm_im=T.Normalize(mean=meani,std=stdi)
        image_n=norm_im(imaget)

        padding=T.Resize(size=(256,256))
        image_nr=padding(image_n)

        return image_nr, image, imgp
    

