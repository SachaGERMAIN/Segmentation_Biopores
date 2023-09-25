import torch
import torch.nn as nn
from dataset_gal import inf_Dataset
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm 
from time import sleep
import matplotlib as mpl


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH = r"C:\Users\ccaurel\Desktop\Python_Deep_Learning\Galleries\models\test_model_1.pth"

IMAGE_PATH ="./data/processed/galleries/image"

inference_data=inf_Dataset(IMAGE_PATH)
inf_loader=DataLoader(inference_data, batch_size=3, shuffle=True)



criterion = nn.CrossEntropyLoss()


model=torch.load(PATH)
model.cuda()
model.eval()

with tqdm(inf_loader) as loop:
    for images, xray,_ in loop:  

        images=images.to(device=device)
        xray=torch.squeeze(xray,dim=1).numpy()
        # Forward pass

        outputs = model(images)

        with torch.no_grad():

            image=outputs.cpu()
            image = image.argmax(dim=1).numpy()
            colormaps=ListedColormap(['darkorange','gold','lawngreen']).resampled(256)
            f,axes=plt.subplots(3,2,figsize=(10, 10))      
            axes[0,0].pcolormesh(image[0],cmap=colormaps,rasterized=True)
            axes[0,0].set_title('Prediction')
            axes[0,0].axis('square')
            axes[0,1].imshow(xray[0],cmap='gray',origin='lower')
            axes[0,1].set_title('Original Image')
            axes[0,1].axis('square')
            axes[1,0].pcolormesh(image[1],cmap=colormaps,rasterized=True)
            axes[1,0].axis('square')
            axes[1,1].imshow(xray[1],cmap='gray',origin='lower')
            axes[1,1].axis('square')
            axes[2,0].pcolormesh(image[2],cmap=colormaps,rasterized=True)
            axes[2,0].axis('square')
            axes[2,1].imshow(xray[2],cmap='gray',origin='lower')
            axes[2,1].axis('square')
            
            plt.show(block=True)
