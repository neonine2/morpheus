import os
import numpy as np
import re
import pandas as pd
import tensorflow as tf

def initialize_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('using GPUs')
        # Restrict TensorFlow to only use the first GPU
        try:
            # for gpu in gpus:
            #     tf.config.experimental.set_virtual_device_configuration(gpu,
            #                             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*21)])
            tf.config.set_visible_devices(gpus[1], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        logical_cpus = tf.config.list_logical_devices('CPU')
        print(len(logical_cpus), "Logical CPU")

def uniquify(path):
    filename, extension = os.path.splitext(path)
    basename = re.sub("\d*$", "", filename)
    counter = 1

    while os.path.exists(path):
        path = basename + "_" + str(counter) + extension
        counter += 1

    return path

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if isinstance(a, pd.DataFrame):
        permA = a.iloc[p]
    else:
        permA = a[p]
    if isinstance(b, pd.DataFrame):
        permB = b.iloc[p]
    else:
        permB = b[p]
    return permA, permB

def sample_cond(x, y, z, eps, train_lb):
    """
    ensure that train_lb% of image patches are in training set and proportion of 
    positive/negative patches are similar across train,test,validation sets
    """
    return (x < eps) and (y < eps) and (z > train_lb)

