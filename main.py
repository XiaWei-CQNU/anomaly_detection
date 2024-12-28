# -*- coding: utf-8 -*-
#
# Author: Wei Xia
# Date:   2024-12-20
#
# main fuction for multivariate time series anomaly detection

from src.parser import args
from train_test.procedure import procedure



if __name__ == '__main__':
    dataset = args.dataset
    less = args.less
    modelname = args.model
    procedure(modelname,dataset,less)
    