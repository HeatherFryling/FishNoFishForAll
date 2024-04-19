# Heather Fryling
# Northeastern University

import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset


class FishLinearImageDataset(Dataset):
    """
    LinearImageDataset
    Defining custom datasets allows us to process jpg, tiff, and exr images in the same way.
    Standard torchvision transforms may force a conversion to PIL.
    Output of this dataset is a float 32 torch tensor scaled to 0,1 and a label for each image.
    """

    def __init__(self, root_dir, custom_transforms=[torch.tensor] ): # Default to convert to tensor. Otherwise, can't load data set.
        """
        Args:
            root_dir (string): Path to the root LINEAR image folder.
            custom_transforms (callable, optional): Optional transforms to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.df = self._build_dataset_df()
        self.class_map = {'NoFish' : 0, 'SwedishFish' : 1}
        self.custom_transforms = custom_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0],
                                self.df.iloc[idx, 1])
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # Flag is important.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=image.astype('float') # Pytorch is not compatible with uint16.
        

        label = self.df.iloc[idx, 0]

        if self.custom_transforms:
          for transform in self.custom_transforms:
            image = transform(image)
        
        class_id = torch.tensor(self.class_map[label])
        image = torch.permute(image, (2, 0, 1))
        image = image.type(torch.FloatTensor)
        image = image.div(65535) # Scale 16-bit values to 0-1

        return image, class_id

    def _build_dataset_df(self):
      labels = os.listdir(self.root_dir)
      data = {'label': [], 'fname': []}
      for label in labels:
        for f in os.listdir(os.path.join(self.root_dir, label)):
          data['label'].append(label)
          data['fname'].append(f)
      return pd.DataFrame.from_dict(data)