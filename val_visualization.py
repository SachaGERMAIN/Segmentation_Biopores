import torch
import torch.nn as nn
from dataset_gal import testDataset
import segmentation_models_pytorch as smp
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm 
from time import sleep
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter




def plot_val(colormaps,image,mask,xray,mask_init,Nr=2,Nc=2):
    fig,axs=plt.subplots(Nr,Nc)
    image,mask=image.cpu(),mask.cpu()
    image,mask=image.argmax(dim=1),torch.squeeze(mask,0)
    imagen,maskn=image.numpy()[0],mask.numpy()
    data=[imagen,maskn]       
    axs[0,0].pcolormesh(data[0], cmap=colormaps, rasterized=True)
    axs[0,0].set_title('Prediction')
    axs[0,1].pcolormesh(data[0], cmap=colormaps, rasterized=True)
    axs[0,1].set_title('Colorized Mask')
    axs[1,0].imshow(xray.squeeze(dim=0),cmap='gray',origin='lower')
    axs[1,0].set_title('Original image')
    axs[1,1].imshow(mask_init.squeeze(dim=0),cmap='gray',origin='lower')
    axs[1,1].set_title('Original mask')
    fig.set_size_inches(10,10)
    plt.show()


PATH = r"C:\Users\ccaurel\Desktop\Python_Deep_Learning\Galleries\models\test_model_1.pth"
IMAGE_PATH = './data/processed/galleries/image'
MASK_PATH = './data/processed/galleries/mask'
                                               
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_data=testDataset(img_path=IMAGE_PATH,mask_path=MASK_PATH)
val_loader=DataLoader(val_data,batch_size=1,shuffle=True)

criterion = nn.CrossEntropyLoss()

model=torch.load(PATH)
model.cuda()

model.eval()


with tqdm(val_loader) as tepoch:
    for images, masks, xray, mask_init in tepoch:
        tepoch.set_description(f"Epoch 1")      

        images,masks=images.to(device=device),masks.to(device=device)
        masks1=torch.squeeze(masks,1).long()

        # Forward pass

        outputs = model(images)
        loss = criterion(outputs, masks1)  
        tp, fp, fn, tn = smp.metrics.get_stats(outputs.argmax(dim=1,keepdim=True), masks, mode='multilabel', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")  
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")     

        with torch.no_grad():
            colormaps=ListedColormap(['darkorange','gold','lawngreen']).resampled(256)
            plot_val(colormaps,outputs,masks1,xray,mask_init)
            
        tepoch.set_postfix(loss=loss.item(), iou=iou_score.item(), f1_score=f1_score.item())
        sleep(0.1)

