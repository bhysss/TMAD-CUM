import json

import openpyxl
import pandas as pd
import warnings

from scorer import Evaluation

warnings.filterwarnings("ignore")

with open("new_dataset/test_French.json", 'r', encoding='UTF-8') as f:
        load_test = json.load(f)

data_test = pd.DataFrame(load_test)
data_test['LABEL'] = 1

data_test_confusion=data_test.sample(frac=1,random_state=3)
data_test_confusion.reset_index(drop=True, inplace=True)

#加载好了随机种子为3的数据集

with open("temp/low_ppl_test_French_bert.json", 'r', encoding='UTF-8') as f:
    load_dict = json.load(f)
# 加载好了训练集中困惑度弄好的了西班牙语数据集，已经一一对应好了！


for weight in range(101):
    weight = weight / 100
    # 读取json文件,对应一阶，二阶的困惑度

    #读取字典中每个自己
    print("**************************************************")
    result_list=[]
    # 困惑度数据集
    for i in range(len(load_dict)):
        print(i)
        Lf_list=[]
        ppl_all_list=[]
        for s in zip(load_dict[str(i)].keys(),load_dict[str(i)].values()):#取出每次的完整形式
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$",s[0])
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$",s[1])
            #计算困惑度
            ppl_all=weight*s[1][0]+(1-weight)*s[1][1]
            Lf_list.append(s[0])
            ppl_all_list.append(ppl_all)
        # 此时，我们需要一个完整形式--对应一个联合困惑度的series，再进行排序取出对应的完整形式，取联合困惑度最小的值
        print("________________________________________________________")

        temp=pd.Series(ppl_all_list, index=Lf_list)
        temp.sort_values(ascending=True,inplace=True)
        print(temp)
        print("value:",temp.values[0])
        print("index:",temp.index[0])
        result_list.append(temp.index[0])
        print("_________________________________________________________")

        del Lf_list
        del ppl_all_list

    pred_list=[]
    true_list=[]

    for r,d in zip(result_list,list(data_test_confusion["label"])):
        pred_list.append(r)
        true_list.append(d)

    # 先true，再pred
    res=Evaluation(true_list,pred_list).run_Eval()


    wb = openpyxl.load_workbook("第一次缩略语实验结果数据.xlsx")
    sheet = wb['sheet2']

    sheet.append([weight,res[0], res[1], res[2],res[3],res[4],res[5],res[6],res[7],res[8],res[9]])
    wb.save("第一次缩略语实验结果数据.xlsx")

    del res
