import os
from PIL import Image

from torch.utils.data import Dataset

import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        # returns the length of image directory specifying how many entries are there
        return len(self.images)

    def __getitem__(self, index):
        # Get the respective image
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg','_mask.gif'))
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255] = 1.0

        # FileNotFoundError: [Errno 2] No such file or directory: 'Carvana/val_masks/fff9b3a5373f_02.jpg'

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        return image, mask