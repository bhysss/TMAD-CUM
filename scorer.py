import argparse
import json
from collections import defaultdict

def run_evaluation(gold,pred):

    res_list = score_expansion(gold, pred, verbos=True)
    # 精确率，召回率，F1值：
    return res_list

def score_expansion(key, prediction, verbos=False):


    correct = 0
    for i in range(len(key)):
        if key[i] == prediction[i]:
            correct += 1
    acc = correct / len(prediction) #计算出正确率
    # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    expansions = set()

    correct_per_expansion = defaultdict(int)#当key不存在时,返回0
    total_per_expansion = defaultdict(int)
    pred_per_expansion = defaultdict(int)

    for i in range(len(key)):
        expansions.add(key[i])#在set中,添加所有真实的标签数据集,
        total_per_expansion[key[i]] += 1 #当key不存在时,直接返回0
        pred_per_expansion[prediction[i]] += 1
        if key[i] == prediction[i]:#如果当前预测正确了,在指定的正确集合上加一
            correct_per_expansion[key[i]] += 1

    precs = defaultdict(int)#设置当前字典,当key不存在时,直接返回0
    recalls = defaultdict(int)
    # 遍历每个正确的样本
    for exp in expansions:
        # 计算的是某个词的精确率   精确率-->指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例  精确率是 tp / (tp + fp)的比例，其中tp是真正性的数量，fp是假正性的数量
        precs[exp] = correct_per_expansion[exp] / pred_per_expansion[exp] if exp in pred_per_expansion else 1
        # 计算的是某个词的召回率   实际为正的样本中被预测为正的样本所占实际为正的样本的比例。
        recalls[exp] = correct_per_expansion[exp] / total_per_expansion[exp]

    micro_prec = sum(correct_per_expansion.values()) / sum(pred_per_expansion.values())
    micro_recall = sum(correct_per_expansion.values()) / sum(total_per_expansion.values())

    micro_f1 = 2*micro_prec*micro_recall/(micro_prec+micro_recall) if micro_prec+micro_recall != 0 else 0

    macro_prec = sum(precs.values()) / len(precs)
    macro_recall = sum(recalls.values()) / len(recalls)
    macro_f1 = 2*macro_prec*macro_recall / (macro_prec+macro_recall) if macro_prec+macro_recall != 0 else 0

    if verbos:
        print('Accuracy: {:.3%}'.format(acc))
        print('-'*10)
        print('Micro Precision: {:.3%}'.format(micro_prec))
        print('Micro Recall: {:.3%}'.format(micro_recall))
        print('Micro F1: {:.3%}'.format(micro_f1))
        print('-'*10)
        print('Macro Precision: {:.3%}'.format(macro_prec))
        print('Macro Recall: {:.3%}'.format(macro_recall))
        print('Macro F1: {:.3%}'.format(macro_f1))
        print('-'*10)
    # 准确率	精确率	召回率	F1值	宏精确率	宏召回率	宏F1值	微精确率	微召回率	微F1值

    res_list=[acc,macro_prec,macro_recall,macro_f1,macro_prec,macro_recall,macro_f1,micro_prec,micro_recall,micro_f1]

    return res_list

class Evaluation:
    def __init__(self,true_list,pred_list):
        self.true_list=true_list
        self.pred_list=pred_list

    def run_Eval(self):
        res_list = run_evaluation(self.true_list, self.pred_list)
        return res_list