import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import tensorflow as tf
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
    
    if model != 'unet':
        train_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(info_dict['mean'], info_dict['stdev']), 
                                            #   transforms.Lambda(torch.asinh),
                                              transforms.ConvertImageDtype(torch.float),
                                            lambda x: torch.mean(x,dim=(1,2))])
        transform = train_transform
    else:
        train_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(img_size),
                                            transforms.Normalize(info_dict['mean'], info_dict['stdev']),
                                            transforms.Lambda(torch.asinh),
                                            transforms.ConvertImageDtype(torch.float),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(degrees=90)])
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(img_size),
                                        transforms.Normalize(info_dict['mean'], info_dict['stdev']),
                                        transforms.Lambda(torch.asinh),
                                        transforms.ConvertImageDtype(torch.float)])
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
    tf.random.set_seed(seed)

def process_path(file_path, IMAGE_SIZE, onlyMean=False):
    label = tf.strings.to_number(tf.strings.split(file_path, os.sep)[-2],tf.int32)
    img = tf.py_function(func=lambda path: np.load(path.numpy().decode("utf-8")),
                        inp=[file_path],
                        Tout=tf.float32)
    img.set_shape(IMAGE_SIZE)
    if onlyMean:
        img = tf.math.reduce_mean(img, axis=(0,1))

    return img, tf.one_hot(label, depth=2)

def make_tf_dataloader(data_path, loader_type, newShape=None, onlyMean=False):

    with open(os.path.join(data_path,'data_info.pkl'), 'rb') as f:
        info_dict = pickle.load(f)
    info_dict['shape'] = (16,16,37)

    mu = info_dict['mean']
    stdev = info_dict['stdev']
    shape = info_dict['shape']
    ds = tf.data.Dataset.list_files(str(os.path.join(data_path,loader_type,'*/*.npy'))).map(lambda x : process_path(x, shape, onlyMean))
    if loader_type == 'train':
        toShuffle = toAugment = toRepeat = True
    else:
        toShuffle = toAugment = toRepeat = False

    if newShape is None:
        newShape = shape
    dataloader = prepare(ds, mu, stdev, shape=newShape, onlyMean=onlyMean, shuffle=toShuffle, augment=toAugment, repeat=toRepeat)
    return dataloader

def make_all_tf_dataloader(output_path, newShape=None, onlyMean=False):
    train_dataloader = make_tf_dataloader(output_path, 'train', newShape, onlyMean=onlyMean)
    validate_dataloader = make_tf_dataloader(output_path, 'validate', newShape, onlyMean=onlyMean)
    test_dataloader = make_tf_dataloader(output_path, 'test', newShape, onlyMean=onlyMean)
    return train_dataloader, validate_dataloader, test_dataloader

def prepare(dataset, mean, stdev, shape=(32,32), onlyMean=False, shuffle=False, augment=False, repeat=False, 
            resize=True, normalize=True, BATCH_SIZE=64, buffer_size=1000, NUM_EPOCH=50, AUTOTUNE=tf.data.AUTOTUNE):
    
    if not onlyMean:
        # Use data augmentation only on the training set.
        if augment:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.5)])
            dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                                num_parallel_calls=AUTOTUNE)
        if resize:
            resize_fn = tf.keras.layers.Resizing(shape[0], shape[1], interpolation='nearest')
            dataset = dataset.map(lambda x, y: (resize_fn(x, training=True), y))

    if normalize:
        normalize_fn = tf.keras.layers.Normalization(axis=-1, mean=mean, variance=stdev**2)
        dataset = dataset.map(lambda x, y: (normalize_fn(x, training=True), y))

    # Randomly shuffle data
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    # Batch all datasets.
    dataset = dataset.batch(BATCH_SIZE)

    # Repeat dataset
    if repeat:
        dataset = dataset.repeat(NUM_EPOCH)

    # Use buffered prefetching on all datasets.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

