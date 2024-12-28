# -*- coding: utf-8 -*-
#
# Author: Wei Xia
# Date:   2024-12-20
#
# 模型测试过程
import torch
import torch.nn as nn
from src.parser import args
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

def convert_to_windows(data, model):
    windows = []; w_size = model.n_window
    #data=[[1],[2],[3],[4],[5]],n_window=3时
    #windows = [[1,1,1],[1,1,2],[1,2,3],[2,3,4],[3,4,5]]
    for i, g in enumerate(data): #enumerate枚举
        if i >= w_size: w = data[i-w_size:i]
        else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    return torch.stack(windows)

def test_dataset_backprop(model, data):
    l = nn.MSELoss(reduction = 'none')
    if model.name == "Bet_In_Chans" :
        l = nn.MSELoss('none')
        #reduction = 'none'返回每个样本的损失
        #reduction = 'mean'返回所有样本的损失均值
        #reduction = 'sum'返回所有样本的损失之和
        l = nn.MSELoss(reduction = 'none')#损失函数 均方误差
        #reduction = 'none'返回每个样本的损失
        #reduction = 'mean'返回所有样本的损失均值
        #reduction = 'sum'返回所有样本的损失之和
        data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
        bs = model.batch
        dataloader = DataLoader(dataset, batch_size = bs, drop_last=True)#batch_size一次迭代中训练样本的数据
        for elem, _ in dataloader:
            # pdb.set_trace()
            # elem = torch.unsqueeze(elem, dim=0)
            z,memory1,memory2,_,_ = model(elem)#1 128  25   1  128  25
            # pdb.set_trace()
            elem = torch.unsqueeze(elem, dim=0)
            z = torch.unsqueeze(z, dim=0)
            if isinstance(z, tuple): 
                z = z[1]
            try:
                loss = torch.cat((loss, l(z, elem)[0]), dim=0) 
                pred_z = torch.cat((pred_z, z[0]), dim=0)
            except:
                loss = l(z, elem)[0]
                pred_z = z[0]
        return loss.detach().numpy(), pred_z.detach().numpy() #均方差损失     重建值。
    elif model.name == "In_Chans":
        l = nn.MSELoss('none')
        #reduction = 'none'返回每个样本的损失
        #reduction = 'mean'返回所有样本的损失均值
        #reduction = 'sum'返回所有样本的损失之和
        l = nn.MSELoss(reduction = 'none')#损失函数 均方误差
        #reduction = 'none'返回每个样本的损失
        #reduction = 'mean'返回所有样本的损失均值
        #reduction = 'sum'返回所有样本的损失之和
        data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
        bs = model.batch
        dataloader = DataLoader(dataset, batch_size = bs, drop_last=True)#batch_size一次迭代中训练样本的数据
        for elem, _ in dataloader:
            # pdb.set_trace()
            # elem = torch.unsqueeze(elem, dim=0)
            z = model(elem)#1 128  25   1  128  25
            # pdb.set_trace()
            elem = torch.unsqueeze(elem, dim=0)
            z = torch.unsqueeze(z, dim=0)
            if isinstance(z, tuple): 
                z = z[1]
            try:
                loss = torch.cat((loss, l(z, elem)[0]), dim=0) 
                sum_z = torch.cat((sum_z, z[0]), dim=0)
            except:
                loss = l(z, elem)[0]
                sum_z = z[0]
        return loss.detach().numpy(), sum_z.detach().numpy(),_,_#均方差损失
    elif model.name == "Bet_Chans":
        l = nn.MSELoss('none')
        #reduction = 'none'返回每个样本的损失
        #reduction = 'mean'返回所有样本的损失均值
        #reduction = 'sum'返回所有样本的损失之和
        l = nn.MSELoss(reduction = 'none')#损失函数 均方误差
        #reduction = 'none'返回每个样本的损失
        #reduction = 'mean'返回所有样本的损失均值
        #reduction = 'sum'返回所有样本的损失之和
        data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
        bs = model.batch
        dataloader = DataLoader(dataset, batch_size = bs, drop_last=True)#batch_size一次迭代中训练样本的数据
        for elem, _ in dataloader:
            # pdb.set_trace()
            # elem = torch.unsqueeze(elem, dim=0)
            z = model(elem)#1 128  25   1  128  25
            # pdb.set_trace()
            elem = torch.unsqueeze(elem, dim=0)
            z = torch.unsqueeze(z, dim=0)
            if isinstance(z, tuple): 
                z = z[1]
            try:
                loss = torch.cat((loss, l(z, elem)[0]), dim=0) 
                sum_z = torch.cat((sum_z, z[0]), dim=0)
            except:
                loss = l(z, elem)[0]
                sum_z = z[0]
        return loss.detach().numpy(), sum_z.detach().numpy(),_,_#均方差损失     预测值02。
    elif model.name == "TranAD":
        l = nn.MSELoss(reduction = 'none')#损失函数 均方误差
        data = convert_to_windows(data, model)
        #reduction = 'none'返回每个样本的损失
        #reduction = 'mean'返回所有样本的损失均值
        #reduction = 'sum'返回所有样本的损失之和
        data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
        bs = model.batch
        dataloader = DataLoader(dataset, batch_size = bs, drop_last=True)#batch_size一次迭代中训练样本的数据
        for d, _ in dataloader:
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, bs, model.n_feats)
            z = model(window, elem)
            if isinstance(z, tuple): z = z[1]#论文中的O2
            try:
                loss = torch.cat((loss, l(z, elem)[0]), dim=0) 
                pred_z = torch.cat((pred_z, z[0]), dim=0)
            except:
                loss = l(z, elem)[0]
                pred_z = z[0]
        return loss.detach().numpy(), pred_z.detach().numpy()#均方差损失     重建值。
    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        return loss.detach().numpy(), y_pred.detach().numpy(),'',''