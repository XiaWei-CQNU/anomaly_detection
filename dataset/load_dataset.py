# -*- coding: utf-8 -*-
#
# Author: Wei Xia
# Date:   2024-12-20
#
# Dadeset loading for multivariate time series anomaly detection

import os
import numpy as np
from torch.utils.data import DataLoader

dataset_folder = 'dataset\processed' #Path to store the Processed dataset

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def cut_array(percentage, arr):
    '''
    Training with small amounts of data
    percentage:The proportion of data
    arr       :All data
    '''
    print(f'{color.BOLD}Slicing dataset to {int(percentage*100)}%{color.ENDC}')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window : mid + window, :]

def load_dataset(dataset, less):
    folder = os.path.join(dataset_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        if dataset == 'UCR': file = '136_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]

    if less: loader[0] = cut_array(0.2, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels

if __name__=='__main__':
    load_dataset('SMD')