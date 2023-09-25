
import torch    
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from dataset_gal import inf_Dataset
from time import sleep
from tqdm import tqdm
from torchvision.utils import save_image
import tifffile as tiff

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = r"C:\Users\ccaurel\Desktop\Python_Deep_Learning\Galleries\models\test_model_1.pth"
SAVE_PATH=r'C:\Users\ccaurel\Desktop\Python_Deep_Learning\data\processed\colonnes\test_without_crop_C1-S8_0_4_V80U_3_0002'
IMAGE_PATH = r'C:\Users\ccaurel\Desktop\Python_Deep_Learning\data\raw\galleries\colonnes\test_without_crop_C1-S8_0_4_V80U_3_0002'


colonne_data=inf_Dataset(IMAGE_PATH)

col_loader=DataLoader(colonne_data, batch_size=10, shuffle=False)

model=torch.load(MODEL_PATH)

model.to(device=device)
model.eval()

colonne=[]
n=1

for image,_,path in tqdm(col_loader):
    image=image.to(device=device)
    output=model(image)
    output = output.cpu().argmax(dim=1).numpy().astype('int32')
    colonne.append(output)
    colonne.append(path)
    print(f"Batch {n}")

    for i in range(len(colonne[0])):
        path=SAVE_PATH+"\\"+colonne[1][i][-29:-4]+".tif"
        tiff.imsave(path,colonne[0][i])
    colonne=[]



    #colonne[1][1][-69:-40]+"_"+colonne[1][1][-8:-4] slicing name