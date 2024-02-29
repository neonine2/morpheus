import os
import numpy as np
import re
import pandas as pd


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