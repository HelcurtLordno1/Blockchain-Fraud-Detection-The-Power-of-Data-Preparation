import pickle as pkl 
from tqdm import tqdm
import pandas as pd

def load_pkl(filename):
    """load data from a pickle file"""
    with open(filename, 'rb') as file:
        return pkl.load(file)

def save_pkl(data, filename):
    """save data to a pickle file"""
    with open(filename, 'wb') as file:
        pkl.dump(data, file)

def load_txt(file_path):
    data = pd.read_csv(file_path,names = ["account"])
    return list(data.account.values)
            