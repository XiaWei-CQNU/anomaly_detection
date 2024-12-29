import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

import pdb


plt.style.use('fast')
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 7, 4

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter1(name, y_true, y_pred):
    ##文件路径  真实值  预测值
    # if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
    os.makedirs(os.path.join('sava_results', name,'plots'), exist_ok=True)
    pdf = PdfPages(f'sava_results/{name}/plots/output1.pdf')
    y_t, y_p = y_true, y_pred
    fig, (ax1) = plt.subplots(1, 1, sharex=True)#共享X轴
    ax1.set_ylabel('Value')
    ax1.set_title(f'dataset = {name}')
    # if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
    ax1.plot(y_t,color='green', linewidth=0.2, label='True')
    ax1.plot(y_p, '-',color='red', alpha=0.6, linewidth=0.3, label='Predicted')
    pdf.savefig(fig)
    plt.close()
    pdf.close()

def plotter(name, y_true, y_pred, ascore, labels):
    ##文件路径   测试值  预测值O2   测试值与预测值的损失  测试值标签
    if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
    os.makedirs(os.path.join('sava_results', name,'plots'), exist_ok=True)
    pdf = PdfPages(f'sava_results/{name}/plots/output.pdf')
    for dim in range(y_true.shape[1]):
        y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)#共享X轴
        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim}')
        # if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
        ax1.plot(smooth(y_t),color='green', linewidth=0.2, label='True')
        ax1.plot(smooth(y_p), '-',color='red', alpha=0.6, linewidth=0.3, label='Predicted')
        ax3 = ax1.twinx()#将ax1的x轴也分配给ax3使用
        ax3.plot(l, '--',color='blue', linewidth=0.3, alpha=0.5)
        ax3.fill_between(np.arange(l.shape[0]), l, color='y', alpha=0.3)
        if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        ax2.plot(smooth(a_s), linewidth=0.2, color='k')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        pdf.savefig(fig)
        plt.close()
    pdf.close()
