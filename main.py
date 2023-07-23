import numpy as np
import pandas as pd
import torch
from pandas import Series
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM

import json
import copy
import gc



device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.no_grad():
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
    model.to(device)
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# 一阶困惑度
def bert_ppl(sent):
    tokenize_input = tokenizer.tokenize(sent)
    # 句子长度
    sen_len = len(tokenize_input)
    # 如果句子长度小于100
    if sen_len < 100:
        tensor_input = tokenizer.convert_tokens_to_ids(tokenize_input)
        # 句子损失为0
        sent_loss = 0.
        all_input = []
        for i, word in enumerate(tensor_input):
            text = copy.copy(tensor_input)
            text[i] = 103  # @@根据具体的BERT模型换???啥意思？
            all_input.append(text)
        all_input = torch.tensor(all_input).to(device)
        output = model(all_input)
        # 应该是预测出来的值
        pred_scores = output[0]
        # print(pred_scores.shape)
        index1 = torch.tensor([[_] for _ in range(pred_scores.shape[0])])
        index2 = torch.tensor([[_] for _ in tensor_input])

        probs = pred_scores[index1, index1].squeeze(1)
        probs = torch.log_softmax(probs, dim=1)

        word_loss = probs[index1, index2]
        sent_loss = torch.sum(word_loss).item()

        ppl = np.exp(-sent_loss / sen_len)
        # 清空当前数值
        del tokenize_input, tensor_input, all_input, output, pred_scores, index1, index2, probs, word_loss, sent_loss
        gc.collect()
        return ppl

    else:
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        sent_loss = 0.
        for i, word in enumerate(tokenize_input):
            tokenize_input[i] = tokenizer.mask_token
            mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
            # print("*************************")
            # print(mask_input.size())
            output = model(mask_input)
            pred_scores = output[0]
            ps = torch.log_softmax(pred_scores[0, i], dim=0)
            word_loss = ps[tensor_input[0, i]]
            sent_loss += word_loss.item()
            tokenize_input[i] = word  # restore
        ppl = np.exp(-sent_loss / sen_len)
        return ppl


# 二阶困惑度
def bert_ppl_second_order(sent):
    tokenize_input = tokenizer.tokenize(sent)
    sen_len = len(tokenize_input)

    if sen_len < 100:

        # 编码
        tensor_input = tokenizer.convert_tokens_to_ids(tokenize_input)
        sent_loss = 0.
        all_input = []

        # 取到句子的倒数第二个
        # 每次掩码两个词
        for i, word in enumerate(tensor_input[:-1]):
            text = copy.copy(tensor_input)
            # print(tokenizer.mask_token)
            text[i] = 103  # @@根据具体的BERT模型换
            text[i + 1] = 103  # @@根据具体的BERT模型换
            all_input.append(text)

        all_input = torch.tensor(all_input).to(device)

        output = model(all_input)
        pred_scores = output[0]

        # [[0], [1], [2], [3], ..., [n]]
        index1 = torch.tensor([[_] for _ in range(pred_scores.shape[0])])

        # [[0, 1], [1, 2], ...]
        index2 = []
        for _ in range(pred_scores.shape[0]):
            index2.append(
                [_, _ + 1]
            )
        index2 = torch.tensor(index2)

        # 获取掩码位置的概率
        probs = pred_scores[index1, index2].squeeze(1)

        probs_new = []
        for _ in range(sen_len):

            # 第一个token，只掩码过一次
            if _ == 0:
                probs_new.append(
                    probs[_, 0,]
                )
            # 最后一个token，只掩码过一次
            elif _ == (sen_len - 1):
                probs_new.append(
                    probs[_ - 1, 1,]
                )
            # 其他token，被掩码过两次，当前的第一个掩码位置以及上一次的后一个掩码位置
            else:
                tmp = probs[_, 0,] + probs[_ - 1, 1,]
                tmp = tmp / 2
                probs_new.append(tmp)

        probs_new = torch.tensor([item.cpu().detach().numpy() for item in probs_new]).to(device)
        probs = torch.log_softmax(probs_new, dim=1)

        # [[0], [1], [2], [3], ..., [n]]
        index3 = torch.tensor([[_] for _ in range(probs.shape[0])])

        # 每个位置对应的正确token的ID
        index4 = torch.tensor([[_] for _ in tensor_input])
        word_loss = probs[index3, index4]
        sent_loss = torch.sum(word_loss).item()
        ppl = np.exp(-sent_loss / sen_len)

        del tokenize_input, tensor_input, all_input, output, pred_scores, index1, index2, probs, word_loss, sent_loss, index3, index4, probs_new
        gc.collect()
        return ppl

    else:
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        sent_loss = 0.
        for i, word in enumerate(tokenize_input[:-1]):

            next_word = tokenize_input[i + 1]

            tokenize_input[i] = tokenizer.mask_token
            tokenize_input[i + 1] = tokenizer.mask_token

            mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
            output = model(mask_input)
            pred_scores = output[0]

            # 当前位置的概率向量
            tmp = pred_scores[0, i]

            if i > 0:
                tmp = (tmp + next) / 2

            ps = torch.log_softmax(tmp, dim=0)

            word_loss = ps[tensor_input[0, i]]
            sent_loss += word_loss.item()

            # 当前位置的下一个位置的概率向量
            next = pred_scores[0, i + 1]

            # 倒数第二个的时候，已经计算了倒数第一个，所以直接得到最后一个的向量并计算
            if i == (sen_len - 2):
                ps = torch.log_softmax(next, dim=0)
                word_loss = ps[tensor_input[0, i + 1]]
                sent_loss += word_loss.item()

            tokenize_input[i] = word  # restore
            tokenize_input[i + 1] = next_word
        ppl = np.exp(-sent_loss / sen_len)
        return ppl

# 三阶困惑度
def bert_ppl_third_order(sent):
    # 输入一句话，先进行分词操作
    tokenize_input = tokenizer.tokenize(sent)
    # 取出句子长度
    sen_len = len(tokenize_input)

    # 当token分词总长度小于100时
    if sen_len < 100:

        # 编码操作，将分词结果转换成词汇表对应的token_id
        tensor_input = tokenizer.convert_tokens_to_ids(tokenize_input)
        sent_loss = 0.
        all_input = []

        # 取到句子的倒数第三个
        # 每次掩码三个词
        for i, word in enumerate(tensor_input[:-2]):
            text = copy.copy(tensor_input)
            # print(tokenizer.mask_token)
            text[i] = 103  # @@根据具体的BERT模型换
            text[i + 1] = 103  # @@根据具体的BERT模型换
            text[i + 2] = 103  # @@根据具体的BERT模型换
            all_input.append(text)
        # 传入到设备中去
        all_input = torch.tensor(all_input).to(device)
        # 从模型中得到所有MASK词向量
        output = model(all_input)
        # 从模型中得到所有MASK词向量的预测结果
        pred_scores = output[0]
        
        # 得到[[0], [1], [2], [3], ..., [n]]的索引下标
        index1 = torch.tensor([[_] for _ in range(pred_scores.shape[0])])

        # [[0, 1, 2], [1, 2, 3], ...]
        index2 = []
        for _ in range(pred_scores.shape[0]):
            index2.append(
                [_, _ + 1,_ + 2]
            )
        index2 = torch.tensor(index2)

        # 获取得到当前所有的三个一组的MASK词向量
        probs = pred_scores[index1, index2].squeeze(1)

        probs_new = []

        # 根据公式来写！
        for _ in range(sen_len):

            # 第一个token，只掩码过一次
            if _ == 0:
                probs_new.append(
                    probs[_, 0,]
                )
            # 最后一个token，只掩码过一次 已经到最后,50
            elif _ == (sen_len - 1):
                probs_new.append(
                    probs[_ - 2, 1,]
                )
            # 第一个掩码有两次的情况
            elif _ == 1:
                tmp_1 = probs[_, 0,] + probs[0, _,]
                tmp_1 = tmp_1 / 2
                probs_new.append(tmp_1)

            # 第二个掩码有两次的情况
            elif _ == (sen_len - 2):  # 49
                tmp_2 = probs[_ - 2, 2,] + probs[_ - 1, 1,]
                tmp_2 = tmp_2 / 2
                probs_new.append(tmp_2)

            # 其他token，被掩码过三次，当前的第一个掩码位置以及上一次的后一个掩码位置
            else:
                tmp = probs[_, 0,] + probs[_ - 1, 1,] + probs[_ - 2, 2]
                tmp = tmp / 3
                probs_new.append(tmp)

        probs_new = torch.tensor([item.cpu().detach().numpy() for item in probs_new]).to(device)
        probs = torch.log_softmax(probs_new, dim=1)#求出log_softmax

        # [[0], [1], [2], [3], ..., [n]]
        index3 = torch.tensor([[_] for _ in range(probs.shape[0])])

        # 原来每个位置对应的token_ID
        index4 = torch.tensor([[_] for _ in tensor_input])

        word_loss = probs[index3, index4]
        sent_loss = torch.sum(word_loss).item()
        ppl = np.exp(-sent_loss / sen_len)

        del tokenize_input, tensor_input, all_input, output, pred_scores, index1, index2, probs, word_loss, sent_loss, index3, index4, probs_new
        gc.collect()
        return ppl
    else:
        # 把所有token对应的id传入到tensor_input中
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        sent_loss = 0.
        # 运行截止到倒数第三个，就可以已经结束
        for i, word in enumerate(tokenize_input[:-2]):
            # 记录填充的词，这一步是为了恢复
            next_word = tokenize_input[i + 1]
            next_next_word = tokenize_input[i + 2]
            # 每次掩码三个词
            tokenize_input[i] = tokenizer.mask_token
            tokenize_input[i + 1] = tokenizer.mask_token
            tokenize_input[i + 2] = tokenizer.mask_token
            mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
            output = model(mask_input)
            # 已经生成了三个对应的MASK词向量结果
            pred_scores = output[0]#生成对应的词向量！

            # 生成当前位置的MASK概率向量
            tmp = pred_scores[0, i]

            if i == 1: #如果当前等于1的话，就要考虑把前两个MASK词向量的结果，给相加并求和并除2
                tmp = (tmp + next) / 2

            if i > 1:#如果当前大于1，则就要考虑把三个相加求和，取平均
                tmp = (tmp + next +next_next ) / 3

            ps = torch.log_softmax(tmp, dim=0)  # 对当前得到的MASK词向量结果取log(softmax())

            word_loss = ps[tensor_input[0, i]]  # 当前MASK，在指定正确单词上的表示
            sent_loss += word_loss.item()  # 将预测的结果进行累加

            # 得到下一个MASK向量的位置MASK概率向量
            next = pred_scores[0, i + 1]
            # 得到第三个向量的位置MASK概率向量
            next_next = pred_scores[0, i + 2]

            # 倒数第三个的时候，已经计算了倒数第一个和倒数第二个，所以直接结算后两个MASK
            if i == (sen_len - 3):#148是句子长度，-1表示最后一个，-2就表示倒数第二个了
                # 计算倒数第二个
                temp=(next+next_next)/2

                ps = torch.log_softmax(temp, dim=0)
                word_loss = ps[tensor_input[0, i + 1]]#倒数第二个位置结果
                sent_loss += word_loss.item() #c += a 等效于 c = c + a

                #计算倒数第一个
                ps1 = torch.log_softmax(next_next, dim=0)
                word_loss = ps1[tensor_input[0, i + 2]]  # 倒数第一个位置结果
                sent_loss += word_loss.item()

            tokenize_input[i] = word  # restore
            tokenize_input[i + 1] = next_word
            tokenize_input[i + 2] = next_next_word

        ppl = np.exp(-sent_loss / sen_len)

        return ppl



from sklearn import metrics
def bk_metrics(y_true, y_pred):
    """
    :param y_true: 真实值
    :param y_pred: 预测值
    :param type: 预测种类
    :return: 评估指标
    """
    print('实验结果如下......')
    print("************************************************************")
    print('准确率/正确率：',metrics.accuracy_score(y_true, y_pred))
    print('精确率：',metrics.precision_score(y_true,y_pred))
    print('召回率：',metrics.recall_score(y_true,y_pred))  # 不求平均
    print('F1值：',metrics.f1_score(y_true,y_pred))
    print("************************************************************")
    print('宏平均精度：', metrics.precision_score(y_true, y_pred, average='macro'))
    print('宏平均召回率:', metrics.recall_score(y_true, y_pred, average='macro'))
    print('宏平均F1值:', metrics.f1_score(y_true, y_pred, average='macro'))
    print("************************************************************")
    print('微平均精度：', metrics.precision_score(y_true, y_pred, average='micro'))
    print('微平均召回率:', metrics.recall_score(y_true, y_pred, average='micro'))
    print('微平均F1值:', metrics.f1_score(y_true, y_pred, average='micro'))
    print("************************************************************")


def list_trans_json(list1_ture,list2_pred,path):
    if not os.path.exists(path):
        os.mkdir(path)
    list1=[]
    list2=[]

    for i in range(len(list1_ture)):
        dic_temp1 = {}

        dic_temp1["ID"]=str(i+1)
        dic_temp1["label"]=list1_ture[i]
        list1.append(dic_temp1)
        del dic_temp1

    for j in range(len(list2_pred)):
        dic_temp2 = {}

        dic_temp2["ID"]=str(j+1)
        dic_temp2["label"]=list2_pred[j]
        list2.append(dic_temp2)
        del dic_temp2

    str_json_1 = json.dumps(list1, ensure_ascii=False, indent=2)
    str_json_2 = json.dumps(list2, ensure_ascii=False, indent=2)

    with open(path+"/true.json", "w", encoding='utf-8') as f:
        f.write(str_json_1)

    with open(path+"/pred.json", "w", encoding='utf-8') as f:
        f.write(str_json_2) 



if __name__ == '__main__':

    
    with open("new_dataset/test_Eng_Le.json", 'r', encoding='UTF-8') as f:
        load_test = json.load(f)

    data_test = pd.DataFrame(load_test)
    data_test['LABEL'] = 1
    
    
    # data_test['label']=1
    print(data_test)
    #先把测试集句子中的缩略语都换成MASK 
    for i, tu_sample in enumerate(zip(data_test["acronym"], data_test["sentence"])):
        abb = tu_sample[0]
        sen = tu_sample[1]
        data_test.loc[i, "sentence"] = sen.replace(abb, "[MASK]")

        
    # data_sample.to_csv("data_sample.csv",index=False)

    import json

    with open("new_dataset/dic_Eng_Le_low.json", 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    # 输入每一个一句话的样本
    final_predict_result1 = {}
    final_predict_result2 = {}
    final_predict_result3 = {}

    
    # 生成一个适用于存储困惑度的字典
    ppl_dic={}

    # list_def_seq=[]
    # 读取处理后的样本数据

    ds = data_test
    
     #先用于保存一阶和二阶

    for i, tu_sample in enumerate(zip(ds["acronym"], ds["sentence"])):
        temp1_score = {}
        temp2_score = {}
        temp3_score = {}

        abb = tu_sample[0]#一个是缩略语
        sen = tu_sample[1]#一个是句子
        print(i)
        # 完整形式对应的困惑度列表的字典
        lf_ppl={}
        for lf in load_dict[abb]:
            # 进行[MASK]替换工作，将原来的[MASK]都换成完整长形式，目前有一个[MASK]会对应好多种完整形式，每一个完整形式对应一种一阶困惑度！关键问题如何  sen
            # list_def_seq.append(lf)
            # print("对应的原形顺序",lf)
            ppl_list=[]

            sentence = sen.replace("[MASK]", lf)
            
            score1 = bert_ppl(sentence)
            score2 = bert_ppl_second_order(sentence)
            score3 = bert_ppl_third_order(sentence)

            ppl_list.append(score1)
            ppl_list.append(score2)
            ppl_list.append(score3)
            

            
            # 我们需要保存完整的长形式，我们可以得到一阶困惑度
            temp1_score[lf] = score1
            temp2_score[lf] = score2
            temp3_score[lf] = score3

   
            # 句子是按照顺序的，所以我们可以按照顺序生成一个字典，然后根据以k1-v的形式，句子为key，value为一个新的字典，记录k2-v（完整形式为k，value值为一个list，主要记录一阶困惑度，二阶困惑度，以及后面的联合困惑度）
            # 我们需要先生成一个json文件！
            
            y1 = {k: v for k, v in sorted(temp1_score.items(), key=lambda item: item[1])}
            y2 = {k: v for k, v in sorted(temp2_score.items(), key=lambda item: item[1])}
            y3 = {k: v for k, v in sorted(temp3_score.items(), key=lambda item: item[1])}
            
            fianl1 = Series(y1)
            fianl2 = Series(y2)
            fianl3 = Series(y3)

            print("*********一阶**************")
            print(fianl1)
            print("--------二阶---------------")
            print(fianl2)
            print("￥￥￥￥￥三阶￥￥￥￥￥￥￥")
            print(fianl3)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$")

            

            # 取出第一个，确定完整形式,我们需要把每一个样本中，它所对应的多种完整形式生成的新样本的句子困惑度，进行直接的保存，一阶，二阶，然后再方便计算出联合困惑度
            
            definitoion1 = fianl1.index[0]
            definitoion2 = fianl2.index[0]
            definitoion3 = fianl3.index[0]

            # 进行一句话一个完整形式的比对，我们只需要得到id和对应的完整形式就可以
            final_predict_result1[i] = definitoion1
            final_predict_result2[i] = definitoion2
            final_predict_result3[i] = definitoion3

            lf_ppl[lf]=ppl_list #加载给字典！

            del ppl_list
            
        ppl_dic[i]=lf_ppl #一个样本句子为索引
        del lf_ppl

        # print("打印出字典结果：")
        # print(ppl_dic)
    print("********************************done******************************************")
    
    # 第四步，进行比对评估，直接和原始正确样本进行对比 正确率，召回率，F值

    list_true1=[]
    list_pred1=[]
    
    list_true2=[]
    list_pred2=[]
    
    list_true3=[]
    list_pred3=[]
    import os
    for index, sample in zip(final_predict_result1.keys(), final_predict_result1.values()):
        # 一个是id，另一个是对应的完整形式
        # if ds["Definition"][index] == sample:
        #     y_pred1.append(1)
        # else:
        #     y_pred1.append(0)
        list_true1.append(ds["label"][index])
        list_pred1.append(sample)

    list_trans_json(list_true1,list_pred1,"一阶法律英语")
        
    print("***************法律英语_bert+一阶困惑度done************************")
    


    for index, sample in zip(final_predict_result2.keys(), final_predict_result2.values()):
        # 一个是id，另一个是对应的完整形式         想办法如何才能取出一阶和二阶的困惑度数值，保存下来！
        # if ds["Definition"][index] == sample:
        #     y_pred2.append(1)
        # else:
        #     y_pred2.append(0)
        list_true2.append(ds["label"][index])
        list_pred2.append(sample)

    list_trans_json(list_true2,list_pred2,"二阶法律英语")

    print("***************法律英语_bert+二阶困惑度done************************")
    


    for index, sample in zip(final_predict_result3.keys(), final_predict_result3.values()):
        # 一个是id，另一个是对应的完整形式         想办法如何才能取出一阶和二阶的困惑度数值，保存下来！
        # if ds["Definition"][index] == sample:
        #     y_pred2.append(1)
        # else:
        #     y_pred2.append(0)
        list_true3.append(ds["label"][index])
        list_pred3.append(sample)

    list_trans_json(list_true3,list_pred3,"三阶法律英语")

    print("***************法律英语_bert+三阶困惑度done************************")
    
    
    
    
    print("###一阶结果############################################")
    from scorer import Evaluation
    res=Evaluation(list_true1,list_pred1).run_Eval()
    print(res[0], res[1], res[2],res[3],res[4],res[5],res[6],res[7],res[8],res[9])
    
    print("###二阶结果############################################")
    from scorer import Evaluation
    res2=Evaluation(list_true2,list_pred2).run_Eval()
    print(res2[0], res2[1], res2[2],res2[3],res2[4],res2[5],res2[6],res2[7],res2[8],res2[9])
    print("###三阶结果############################################")
    from scorer import Evaluation
    res3=Evaluation(list_true3,list_pred3).run_Eval()
    print(res3[0], res3[1], res3[2],res3[3],res3[4],res3[5],res3[6],res3[7],res3[8],res3[9])

    


