import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

# Hyperparams
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = 'Carvana/train_images'
TRAIN_MASK_DIR = 'Carvana/train_masks'
VAL_IMG_DIR = 'Carvana/val_images'
VAL_MASK_DIR = 'Carvana/val_masks'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx , (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, 
        TRAIN_MASK_DIR, 
        VAL_IMG_DIR, 
        VAL_MASK_DIR, 
        BATCH_SIZE, 
        train_transform, 
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict() 
        }

        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder='saved_imgs/', device=DEVICE
        )

if __name__ == '__main__':
    main()


'''
n/Research-Paper-Pytorch-Implementations/research_impl/bin/python /home/kevin/Research-Paper-Pytorch-Implementations/unet_pytorch/train.py
/home/kevin/Research-Paper-Pytorch-Implementations/research_impl/lib/python3.9/site-packages/torch/cuda/__init__.py:143: UserWarning: 
NVIDIA GeForce RTX 3060 Laptop GPU with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the NVIDIA GeForce RTX 3060 Laptop GPU GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
Traceback (most recent call last):
  File "/home/kevin/Research-Paper-Pytorch-Implementations/unet_pytorch/train.py", line 125, in <module>
    main()
  File "/home/kevin/Research-Paper-Pytorch-Implementations/unet_pytorch/train.py", line 87, in main
    train_loader, val_loader = get_loaders(
  File "/home/kevin/Research-Paper-Pytorch-Implementations/unet_pytorch/utils.py", line 26, in get_loaders
    train_ds = CarvanaDataset(
  File "/home/kevin/Research-Paper-Pytorch-Implementations/unet_pytorch/dataset.py", line 13, in __init__
    self.images = os.listdir(image_dir)
FileNotFoundError: [Errno 2] No such file or directory: 'Carvana/train_images'
'''