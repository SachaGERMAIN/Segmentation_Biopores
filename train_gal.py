import segmentation_models_pytorch as smp
import torch    
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
from dataset_gal import testDataset
from time import sleep
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#writer = SummaryWriter()


PATH = r"C:\Users\ccaurel\Desktop\Python_Deep_Learning\Galleries\models\test_model_1.pth"
IMAGE_PATH = './data/processed/galleries/image'
MASK_PATH = './data/processed/galleries/mask'
 
num_classes = 3
num_epochs = 60                  
batch_size = 12
learning_rate = 0.001


### Model definition and pretraining encoders and assigning to gpu

model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)

model.cuda()

preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', "imagenet")

# Dataset opening and initialization of the DataLoader

data=testDataset(img_path=IMAGE_PATH,mask_path=MASK_PATH)
train_data,val_data=random_split(data,lengths=[0.8,0.2])

train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=True)

# Loss and optimizer definition

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 


#### Training loop

n_ep=0


for epoch in range(num_epochs):
    model.train()
    train_loss=0
    loop_train=tqdm(train_loader,total = len(train_data)/batch_size,disable=True,)
    for images, masks,_ ,_ in loop_train:
            sleep(0.01)
            
            # Tweaking mask format : mask goes from 1*1*256*256 to 1*256*256, and assigning both image and mask to gpu for computation
            
            images,masks=images.to(device=device),masks.to(device=device)
            masks1=torch.squeeze(masks,1).long()

            # Forward pass

            outputs = model(images)

            loss = criterion(outputs, masks1) 

            # Backward pass and optimize

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

    print(f"mean batch loss : {train_loss/len(train_loader)}")
    print(f"Epoch train {epoch+1} done")

    val_loss=0
    val_iou=0
    val_F1=0
    n=0

    model.eval()   

    loop_val=tqdm(val_loader,total = len(val_data)/batch_size)
    for images, masks,_,_ in loop_val:

        images,masks=images.to(device=device),masks.to(device=device)
        masks1=torch.squeeze(masks,1).long()

        output = model(images)
        loss = criterion(output,masks1)
        val_loss += loss.item() 

        # Scores calculations

        tp, fp, fn, tn = smp.metrics.get_stats(output.argmax(dim=1,keepdim=True), masks, mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")  
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        val_iou+=iou_score.item()
        val_F1+=f1_score.item()
        n+=1
    print(f"mean batch loss : {val_loss/len(val_loader)}, mean IoU : {val_iou/len(val_loader)}, mean F1 : {val_F1/len(val_loader)}")
    print(f"Epoch val {epoch+1} done")


    # add to tensorboard

    # writer.add_scalar("Loss/train", loss.item(), epoch)
    # writer.add_scalar("Loss/val", val_loss/len(val_loader), epoch)
    # writer.add_scalar("IoU/val", val_iou/len(val_loader), epoch)           
    # writer.add_scalar("F1_score/val", val_F1/len(val_loader), epoch)



    n_ep+=1

# writer.flush()
# writer.close()

# saving the entire model, could be only the state_dir for the trained weights

torch.save(model, PATH)
