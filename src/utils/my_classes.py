import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle

class torchDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, 'label.csv'))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        # Get data and label   
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, f'{label}/patch_{idx}.npy')
        image = np.load(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    
def make_torch_dataloader(data_path, img_size, model='mlp',
                          params = {'batch_size': 64,'num_workers': 4,'pin_memory':True}):
    with open(os.path.join(data_path,'data_info.pkl'), 'rb') as f:
        info_dict = pickle.load(f)
    
    transformation = [transforms.ToTensor(),
                      transforms.Normalize(info_dict['mean'], info_dict['stdev']),
                      transforms.ConvertImageDtype(torch.float)]
    if model != 'unet':
        train_transform = transforms.Compose(transformation+
                                            [lambda x: torch.mean(x,dim=(1,2))])
        transform = train_transform
    else:
        train_transform = transforms.Compose(transformation+
                                            [transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(degrees=90)])
        transform = transforms.Compose(transformation)
    training_data = torchDataset(data_path + '/train', transform=train_transform)
    validation_data = torchDataset(data_path + '/validate', transform=transform)
    testing_data = torchDataset(data_path + '/test', transform=transform)

    train_loader = DataLoader(training_data, shuffle= True, **params)
    val_loader = DataLoader(validation_data, shuffle= False, **params)
    test_loader = DataLoader(testing_data, shuffle= False, **params)
    return train_loader, val_loader, test_loader

def set_pytorch_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def set_seed(seed):
    np.random.seed(seed)

