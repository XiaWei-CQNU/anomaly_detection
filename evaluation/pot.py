import numpy as np

from evaluation.spot import SPOT
from src.constants import lm
from evaluation.evaluation import evaluation
from evaluation.adjust_predicts import adjust_predicts
from drawing.plotting import plotter1
from src.parser import args



def pot_eval(init_score, score, label, q=1e-5, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """
    lms = lm[0]
    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
        except: lms = lms * 0.999
        else: break
    ret = s.run(dynamic=False)  # run
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds']) * lm[1]
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    # DEBUG - np.save(f'{debug}.npy', np.array(pred))
    # DEBUG - print(np.argwhere(np.array(pred)))
    plotter1(f'{args.model}_{args.dataset}', pred, label)#画图，保存在\plots文件夹下
    p_t = evaluation(pred, label)
    # print('POT result: ', p_t, pot_th, p_latency)
    return {
        'f1': p_t.get_f1(),
        'precision': p_t.get_precision(),
        'recall': p_t.get_recall(),
        'TP': p_t.TP,
        'TN': p_t.TN,
        'FP': p_t.FP,
        'FN': p_t.FN,
        'ROC/AUC': p_t.get_roc_auc(),
        'threshold': pot_th,
        # 'pot-latency': p_latency
    }, np.array(pred).astype(int)