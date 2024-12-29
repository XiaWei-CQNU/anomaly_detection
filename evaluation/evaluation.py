# -*- coding: utf-8 -*-
#
# Author: Wei Xia
# Date:   2024-12-20
#
# 评价指标
# 
import numpy as np
from sklearn.metrics import roc_auc_score

class evaluation:
    def __init__(self, predict_label, actual_label):

        self.TP = np.sum(predict_label * actual_label)
        self.TN = np.sum((1 - predict_label) * (1 - actual_label))
        self.FP = np.sum(predict_label * (1 - actual_label))
        self.FN = np.sum((1 - predict_label) * actual_label)

        self.predict = predict_label
        self.actual = actual_label
        
    def get_precision(self):
        return self.TP / (self.TP + self.FP + 0.00001)
    
    def get_recall(self):
        return self.TP / (self.TP + self.FN + 0.00001)
    
    def get_f1(self):
        return 2/((self.TP + self.FP)/(self.TP  + 0.00001) + (self.TP + self.FN)/(self.TP  + 0.00001))
    
    def get_roc_auc(self):
        try:
            roc_auc = roc_auc_score(self.actual, self.predict)
        except:
            roc_auc = 0
        return roc_auc



