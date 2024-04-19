# Heather Fryling
# Northeastern University

import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset


class FishColorAndIntensityDataset(Dataset):
    """
    FishColorAndIntensityDatset
    Defining custom datasets allows us to process jpg, tiff, and exr images in the same way.
    Standard torchvision transforms may force a conversion to PIL.
    Use the correct transforms with this datset to output linear, sRGB, or log images.
    Either linear_to_srgb (for sRGB), linear_to_float_linear (for linear), or linear_to_log (for log) should be the last transform applied before
    torch.tensor.
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

        return image, class_id

    def _build_dataset_df(self):
      labels = os.listdir(self.root_dir)
      data = {'label': [], 'fname': []}
      for label in labels:
        for f in os.listdir(os.path.join(self.root_dir, label)):
          data['label'].append(label)
          data['fname'].append(f)
      return pd.DataFrame.from_dict(data)