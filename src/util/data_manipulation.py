# Heather Fryling
# Northeastern University

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# Splits the dataset into train and test.
# https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/4
def train_val_dataset(dataset, val_split=0.2):
    '''
    Provides a random train/val split for a Pytorch dataset.
    dataset: a Pytorch dataset
    val_split: the amount of data to allocate to the validation set in range [0, 1]
    '''
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets