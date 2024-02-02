# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit - Fabio Palmisano -*-
# -*- date : 2024-01-20 -*-
# -*- Last revision: 2024-02-02 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- Description: Functions to open and store files in binary format-*-

#import libraries
import pickle

def save_pickle(result, file_path="pickle"):
    """Save a variable in a binary format

    Args:
        result: dataFrame
        file_path: file path where to store this variable

    Returns:
    """
    with open(file_path, "wb") as file:
        pickle.dump(result, file)

def open_pickle(file_path="pickle"):
    """Open a variable in a binary format

    Args:
        file_path: file path where to store this variable

    Returns:
        result: variable stored in the file_path
    """
    with open(file_path, "rb") as file:
        result = pickle.load(file)
    return result

def unpickle(file):
    """Open a variable in a binary format
    Args:
        file: file path where to store this variable
    Returns:
        dict: return diciotnary
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict