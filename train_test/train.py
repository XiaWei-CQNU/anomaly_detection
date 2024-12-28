# -*- coding: utf-8 -*-
#
# Author: Wei Xia
# Date:   2024-12-20
#
# 模型训练过程

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

def train_dataset_backprop(epoch, model, data, optimizer, scheduler):
    l = nn.MSELoss(reduction = 'mean')
    if model.name == "Bet_In_Chans":
        l = nn.MSELoss(reduction = 'none')#损失函数 均方误差
        #reduction = 'none'返回每个样本的损失
        #reduction = 'mean'返回所有样本的损失均值
        #reduction = 'sum'返回所有样本的损失之和
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch
        dataloader = DataLoader(dataset, batch_size = bs, drop_last=True)#batch_size一次迭代中训练样本的数据
        l1s, l2s = [], []
        for elem, _ in dataloader:
            z,_,_,In_attn,Bet_attn = model(elem)#forward
            elem = torch.unsqueeze(elem, dim=0)
            z = torch.unsqueeze(z, dim=0)
            l1 = l(z, elem)
            #此时n=1，(1 - 1/n) * l(z[1], elem)=0，对应公式7中的L1，。
            #isinstance 判断数据是不是指定的类型
            if isinstance(z, tuple): z = z[1]
            l1s.append(torch.mean(l1).item())#计算平均误差
            loss = torch.mean(l1)
            optimizer.zero_grad()#梯度初始化0
            loss.backward(retain_graph=True)#反向传播求梯度
            optimizer.step()#更新参数
        scheduler.step()#调整学习率scheduler
        #根据 epoch 的数量调整学习率。学习率调度应该在优化器更新后应用
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        return np.mean(l1s), optimizer.param_groups[0]['lr']
        # return 平均损失 学习率
        # Pytorch 中优化器 Optimizer 的一个属性，它是一个列表，其中的每个元素都是一个字典，表示优化的参数组。
        # 每个字典都包含了一组参数的各种信息，如当前的学习率、动量等。
        # 这个属性可以用来获取优化器中当前管理的参数组的信息，也可以用来修改优化器的参数设置。
    elif model.name == "In_Chans":
        l = nn.MSELoss(reduction = 'none')#损失函数 均方误差
        #reduction = 'none'返回每个样本的损失
        #reduction = 'mean'返回所有样本的损失均值
        #reduction = 'sum'返回所有样本的损失之和
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch
        dataloader = DataLoader(dataset, batch_size = bs, drop_last=True)#batch_size一次迭代中训练样本的数据
        l1s, l2s = [], []
        for elem, _ in dataloader:
            z = model(elem)#forward
            elem = torch.unsqueeze(elem, dim=0)
            z = torch.unsqueeze(z, dim=0)
            l1 = l(z, elem)
            #此时n=1，(1 - 1/n) * l(z[1], elem)=0，对应公式7中的L1，。
            #isinstance 判断数据是不是指定的类型
            if isinstance(z, tuple): z = z[1]
            l1s.append(torch.mean(l1).item())#计算平均误差
            loss = torch.mean(l1)
            optimizer.zero_grad()#梯度初始化0
            loss.backward(retain_graph=True)#反向传播求梯度
            optimizer.step()#更新参数
        scheduler.step()#调整学习率scheduler
        #根据 epoch 的数量调整学习率。学习率调度应该在优化器更新后应用
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        return np.mean(l1s), optimizer.param_groups[0]['lr']
        # return 平均损失 学习率
        # Pytorch 中优化器 Optimizer 的一个属性，它是一个列表，其中的每个元素都是一个字典，表示优化的参数组。
        # 每个字典都包含了一组参数的各种信息，如当前的学习率、动量等。
        # 这个属性可以用来获取优化器中当前管理的参数组的信息，也可以用来修改优化器的参数设置。
    elif model.name == "TranAD":
        l = nn.MSELoss(reduction = 'none')#损失函数 均方误差
        data = convert_to_windows(data, model)
        #reduction = 'none'返回每个样本的损失
        #reduction = 'mean'返回所有样本的损失均值
        #reduction = 'sum'返回所有样本的损失之和
        data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
        bs = model.batch
        dataloader = DataLoader(dataset, batch_size = bs)#batch_size一次迭代中训练样本的数据
        n = epoch + 1; w_size = model.n_window
        l1s, l2s = [], []
        for d, _ in dataloader:
            local_bs = d.shape[0]#迭代的样本数128
            window = d.permute(1, 0, 2)#128 10 55 ---->10 128 55
            elem = window[-1, :, :].view(1, local_bs, model.n_feats)#1*128*55
            z = model(window, elem)#forward
            l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
            #此时n=1，(1 - 1/n) * l(z[1], elem)=0，对应公式7中的L1，。
            #isinstance 判断数据是不是指定的类型
            if isinstance(z, tuple): z = z[1]
            l1s.append(torch.mean(l1).item())#计算平均误差
            loss = torch.mean(l1)
            optimizer.zero_grad()#梯度初始化0
            loss.backward(retain_graph=True)#反向传播求梯度
            optimizer.step()#更新参数
        scheduler.step()
        #根据 epoch 的数量调整学习率。学习率调度应该在优化器更新后应用
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        return np.mean(l1s), optimizer.param_groups[0]['lr']
        #return 平均损失 学习率
        #Pytorch 中优化器 Optimizer 的一个属性，它是一个列表，其中的每个元素都是一个字典，表示优化的参数组。
        # 每个字典都包含了一组参数的各种信息，如当前的学习率、动量等。
        # 这个属性可以用来获取优化器中当前管理的参数组的信息，也可以用来修改优化器的参数设置。
    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.item(), optimizer.param_groups[0]['lr']
