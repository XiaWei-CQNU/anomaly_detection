# -*- coding: utf-8 -*-
#
# Author: Wei Xia
# Date:   2024-12-20
#
# main fuction for multivariate time series anomaly detection
 
from src.parser import args
from dataset.load_dataset import load_dataset
from model.load_model import load_model
from model.sava_model import save_model
from src.constants import color
from time import time
from tqdm import tqdm
from train_test.train import train_dataset_backprop
from train_test.test import test_dataset_backprop
import torch
from drawing.plotting import plotter
import pandas as pd
import numpy as np
from evaluation.pot import pot_eval
from src.diagnosis import ndcg,hit_att
from pprint import pprint


def procedure(modelname:str,dataset:str,less:bool):
    train_loader, test_loader, labels = load_dataset(dataset, less)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(modelname, labels.shape[1])
    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    divid = labels.shape[0]%model.batch
    if divid !=0:
        labels = labels[:-divid,:]
    
    ### Training phase
    print(f'{color.HEADER}Training {modelname} on {dataset}{color.ENDC}')
    num_epochs = 5; e = epoch + 1; start = time()
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
        lossT, lr = train_dataset_backprop(e, model, trainD, optimizer, scheduler)
        # optimizer, scheduler 优化器（更新参数），调整学习率
        accuracy_list.append((lossT, lr))
    Training_time = "{:.4f}".format((time()-start))+' s'
    Per_Training_time = "{:.4f}".format((time()-start)/num_epochs)+' s'
    print(color.BOLD+'Training time: '+"{:.4f}".format(time()-start)+' s'+color.ENDC)
    save_model(model, optimizer, scheduler, e, accuracy_list)#保存模型
     ### Testing phase
    torch.zero_grad = True
    model.eval()
    # pdb.set_trace()
    # 转换为模型评估模式
    # 如果需要继续进行训练，需要通过调用model.train()将模型切换回训练模式
    print(f'{color.HEADER}Testing {modelname} on {dataset}{color.ENDC}')
    loss, y_pred = test_dataset_backprop(model, testD)
    if not args.test:
        plotter(f'{modelname}_{dataset}', testD[:loss.shape[0],:], y_pred, loss, labels)#画图，保存在\plots文件夹下
        #文件路径   测试值  重建值   测试值与预测值的损失(均方差MSE)  测试集标签

    lossFinal = np.mean(loss, axis=1)
    np.argsort(lossFinal)[0:100]#对异常分数从大到小排序，返回索引信息。

    df2 = pd.DataFrame()
    lossT, _ = test_dataset_backprop(model, trainD)
    #训练集损失，用作确定POT阈值。
    preds = []
    for i in range(loss.shape[1]):
        '''
        检测每一个特征的异常
        '''
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls)#result 单个特征的实验评估结果(ROC,F1,precision....)  pred:单个特征的预测标签
        preds.append(pred.tolist())
        df2 = df2._append(result, ignore_index=True)
    preds = np.array(preds).transpose()#汇总所有特征的预测标签
    # preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
    # pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)#训练集的平均损失 测试集的平均损失
    #pdb.set_trace()
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0#一个时间点存在某个特征异常，将这个时间点定为异常时间点
    # pdb.set_trace()
    result, pred_label_Final = pot_eval(lossTfinal, lossFinal, labelsFinal)#实验评估结果(ROC,F1,precision....)  预测标签

    #保存数据用作分析
    result.update({
                    'Training_time':Training_time,
                    'Per_Training_time':Per_Training_time
                })   
    
    Mid_Data = {
        'ori_data':testD.tolist(),#原始数据
        'pred_data':y_pred.tolist(),#重构数据
        'loss':loss.tolist(),#汇集了所有单个特征的损失 
        'lossFinal':lossFinal.tolist(),#测试集，平均了单个时间点的所有特征损失
        'labels':labels.tolist(),#所有数据的真实标签
        'labelsFinal':labelsFinal.tolist(),#所有数据的预测标签
        'pred_label_Final':pred_label_Final.tolist(),#单个时间点真实标签
        'preds_label':preds.tolist()#单个时间点预测标签
    }

    filename = "sava_results/"+modelname+"_"+dataset+"/Experimental_Results.json"
    # 将数据保存到JSON文件
    with open(filename, 'w') as file:
        import json
        json.dump(Mid_Data, file)

    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    np.argsort(lossFinal)[0:100]#对异常分数从大到小排序，返回索引信息。
    print(df2)
    pprint(result)