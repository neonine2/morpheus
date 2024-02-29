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
