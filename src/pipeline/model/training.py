"""
Resources
- https://github.com/DatumLearning/Mask-RCNN-finetuning-PyTorch/blob/main/notebook.ipynb
- https://www.youtube.com/watch?v=vV9L71hK-RE
- https://www.youtube.com/watch?v=t1MrzuAUdoE

"""

#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


plot_dir = '/home/abuzarmahmood/projects/auto_slide/plots'
artifacts_dir = '/home/abuzarmahmood/projects/auto_slide/artifacts'

img_dir = '/home/abuzarmahmood/projects/auto_slide/data/labelled_images/images/'
mask_dir = '/home/abuzarmahmood/projects/auto_slide/data/labelled_images/masks/'
images = sorted(os.listdir(img_dir))
masks = sorted(os.listdir(mask_dir))


idx = 0
img = Image.open(img_dir + images[idx]).convert("RGB")
mask = Image.open(mask_dir + masks[idx])

fig,axis = plt.subplots(1,2,figsize=(10,5))
axis[0].imshow(img)
axis[1].imshow(mask)
plt.show()


class CustDat(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        self.imgs = images
        self.masks = masks
        # Define augmentation transformations
        self.transform = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor()
        ])
        # Simple transform for validation/testing
        self.base_transform = T.ToTensor()

    def __getitem__(self, idx):
        img = Image.open(img_dir + self.imgs[idx]).convert("RGB")
        mask = Image.open(mask_dir + self.masks[idx])
        mask = np.array(mask) // 255
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        masks = np.zeros((num_objs, mask.shape[0], mask.shape[1]))
        for i in range(num_objs):
            masks[i][mask == i+1] = True
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        
        # Apply augmentation during training
        img_tensor = self.transform(img)
        return img_tensor, target

    def __len__(self):
        return len(self.imgs)


model = torchvision.models.detection.maskrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features , 2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask , hidden_layer , 2)


# No need for a separate transform variable as it's now in the CustDat class

def custom_collate(data):
  return data

num = int(0.9 * len(images))
num = num if num % 2 == 0 else num + 1
train_imgs_inds = np.random.choice(range(len(images)) , num , replace = False)
val_imgs_inds = np.setdiff1d(range(len(images)) , train_imgs_inds)
train_imgs = np.array(images)[train_imgs_inds]
val_imgs = np.array(images)[val_imgs_inds]
train_masks = np.array(masks)[train_imgs_inds]
val_masks = np.array(masks)[val_imgs_inds]

train_dl = torch.utils.data.DataLoader(CustDat(train_imgs , train_masks) ,
                                 batch_size = 2 ,
                                 shuffle = True ,
                                 collate_fn = custom_collate ,
                                 num_workers = 1 ,
                                 pin_memory = True if torch.cuda.is_available() else False)
val_dl = torch.utils.data.DataLoader(CustDat(val_imgs , val_masks) ,
                                 batch_size = 2 ,
                                 shuffle = True ,
                                 collate_fn = custom_collate ,
                                 num_workers = 1 ,
                                 pin_memory = True if torch.cuda.is_available() else False)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

model.to(device)




params = [p for p in model.parameters() if p.requires_grad]




optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


# for i, dt in enumerate(train_dl):
#     imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
#     targ = [dt[0][1] , dt[1][1]]
#     targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
#     break


n_epochs = 30
all_train_losses = []
all_val_losses = []
best_val_loss = float('inf')  # Track the best validation loss
flag = False
for epoch in trange(n_epochs):
    train_epoch_loss = 0
    val_epoch_loss = 0
    model.train()
    n_train = len(train_dl)
    pbar = tqdm(train_dl)
    for i , dt in enumerate(pbar): 
        pbar.set_description(f"Epoch {epoch}/{n_epochs}, Batch {i}/{n_train}")
        imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
        targ = [dt[0][1] , dt[1][1]]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
        loss = model(imgs , targets)
        if not flag:
            print(loss)
            flag = True
        losses = sum([l for l in loss.values()])
        train_epoch_loss += losses.cpu().detach().numpy()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    all_train_losses.append(train_epoch_loss)
    with torch.no_grad():
        for j , dt in enumerate(val_dl):
            if len(dt) < 2:
                continue
            imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
            targ = [dt[0][1] , dt[1][1]]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
            loss = model(imgs , targets)
            losses = sum([l for l in loss.values()])
            val_epoch_loss += losses.cpu().detach().numpy()
        all_val_losses.append(val_epoch_loss)
    print(epoch , "  " , train_epoch_loss , "  " , val_epoch_loss)
    
    # Save model only if validation loss improves
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), os.path.join(artifacts_dir, 'mask_rcnn_model.pth'))
        print(f"New best model saved with validation loss: {best_val_loss}")


# Save loss histories
np.save(artifacts_dir + '/train_losses.npy', all_train_losses)
np.save(artifacts_dir + '/val_losses.npy', all_val_losses)



fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].plot(all_train_losses)
ax[1].plot(all_val_losses)
ax[0].set_title("Train Loss")
ax[1].set_title("Validation Loss")
# plt.show()
fig.savefig(plot_dir + '/train_val_loss.png')
plt.close(fig)


# Plot example predicted mask from validation set
model.eval()

for img_path in tqdm(val_imgs):
    img = Image.open(img_dir + img_path).convert("RGB")
    # Use the base transform for prediction to match training
    transform = T.ToTensor()
    ig = transform(img)
    with torch.no_grad():
        pred = model([ig.to(device)])


    n_preds = len(pred[0]["masks"])
    fig, ax = plt.subplots(1, n_preds+1, figsize=(5*n_preds,5))
    ax[0].imshow(img)
    for i in range(n_preds):
        ax[i+1].imshow((pred[0]["masks"][i].cpu().detach().numpy() * 255).astype("uint8").squeeze())
    # plt.show()
    fig.savefig(plot_dir + f'/{img_path.split(".")[0]}example_masks.png')
    plt.close(fig)


    all_preds = np.stack([(pred[0]["masks"][i].cpu().detach().numpy() * 255).astype("uint8").squeeze() for i in range(n_preds)])


    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(img)
    ax[1].imshow(all_preds.mean(axis = 0))
    # plt.show()
    plt.savefig(plot_dir + f'/{img_path.split(".")[0]}mean_example_mask.png')
    plt.close()






