import random
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, BertTokenizerFast, get_linear_schedule_with_warmup
import pandas
import os
import json
import warnings

"""
bert-base-multilingual-cased
malay-huggingface/bert-base-bahasa-cased
"""
# 拼接式代码
warnings.filterwarnings('ignore')
# 设置所有的随机种子

def seed_everything(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# @title GlobalConfig
class GlobalConfig:
    def __init__(self):
        self.seed = 2022
        self.path = Path('./data/')
        self.max_length = 128
        self.roberta_path = 'bert-base-multilingual-cased'  # @param
        self.num_workers = os.cpu_count()
        self.batch_size = 4
        self.steps_show = 100
        self.accum_steps = 1
        num_epochs = 10  # @param
        self.epochs = num_epochs
        self.warmup_steps = 0
        lr = 5e-6  # @param
        self.lr = lr  # modified from 3e-5
        run_id = "stage1_concat"  # @param
        self.offline = True
        self.saved_model_path = run_id
        self.n_splits = 5


# 模型设置在装置中
def move_to_device(x, device):
  if callable(getattr(x, 'to', None)): return x.to(device)
  if isinstance(x, (tuple, list)): return [move_to_device(o, device) for o in x]
  elif isinstance(x, dict): return {k: move_to_device(v, device) for k, v in x.items()}
  return x

# 数据集
class TweetBertDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_testing = is_testing

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):#两个句子之间的拼接，弄在一起了的输入

        text_sen = self.df.iloc[ix]['Sentence']
        text_def = self.df.iloc[ix]['Definition']

        input_ids_sen = tokenizer.encode(text_sen)
        input_ids_def = tokenizer.encode(text_def)
        # Token embedding 102就是sep
        input_ids_sen = input_ids_sen[:self.max_length-1] + [102]
        input_ids_def = input_ids_def[1:self.max_length] + [102]
        # Position embedding：maskattention需要关注的地方均为1。
        attn_mask = [1] * len(input_ids_sen)
        attn_mask += [1] * len(input_ids_def)
        token_type_ids = [0] * len(input_ids_sen)#segment embedding中只有 0 和 1两个值
        token_type_ids += [1] * len(input_ids_def)

        input_ids = input_ids_sen + input_ids_def

        # PAD
        pad_len = self.max_length * 2 - len(input_ids)
        input_ids += [0] * pad_len
        attn_mask += [0] * pad_len
        token_type_ids += [0] * pad_len

        input_ids, attn_mask, token_type_ids = map(torch.LongTensor,
                                                   [input_ids, attn_mask, token_type_ids])

        encoded_dict = {
            'input_ids1': input_ids,
            'attn_mask1': attn_mask,
            'token_type_ids1': token_type_ids,
        }
        if not self.is_testing:
            # tmp = [0] * 3
            # tmp[all_labels.index(sentiment)] = 1
            # encoded_dict['sentiment'] = torch.tensor(tmp, dtype=torch.long)
            sentiment = self.df.iloc[ix]['label']
            encoded_dict['label'] = torch.tensor(all_labels.index(sentiment), dtype=torch.long)
        return encoded_dict

# 加载预选量模型
class TweetBertModel(nn.Module):
    def __init__(self, roberta_path):
        super().__init__()
        # 声明
        roberta_config = BertConfig.from_pretrained(roberta_path)
        roberta_config.output_hidden_states = True
        # roberta_config.attention_mode = 'sliding_chunks'
        # roberta_config.gradient_checkpointing = True
        # 声明预训练模型
        self.roberta = BertModel.from_pretrained(roberta_path, config=roberta_config)
        # 声明灭活层
        self.dropout = nn.Dropout(0.5)
        # 声明线性分类
        self.classifier = nn.Linear(roberta_config.hidden_size, 2)
        torch.nn.init.normal_(self.classifier.weight, std=0.02)
        # self.soft = nn.Sigmoid()

    def forward(self, input_ids1, attn_mask1, token_type_ids1):
        # 前馈构造

        # 先经过预训练模型的训练
        pooled_output1 = self.roberta(
            input_ids=input_ids1,
            attention_mask=attn_mask1,
            token_type_ids=token_type_ids1
        )['last_hidden_state']

        #开始灭活层进行灭活
        pooled_output1 = self.dropout(pooled_output1[:,0,:])
        # 在进行最终的分类
        start_logits = self.classifier(pooled_output1)

        return start_logits,pooled_output1


# 新增的对抗样本
class FGM():

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 开始对模型进行训练
def train_(train_iter,val_iter,model,fold):
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.AdamW(optimizer_parameters, lr=GCONF.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(GCONF.warmup_steps * train_steps),
        num_training_steps=train_steps
    )
    steps = 0
    best_f1 = 0
    model.train()
    # fgm = FGM(model)
    # g = open('f1.txt','a')
    for epoch in range(1, GCONF.epochs + 1):
        for batch in train_iter:
            input_ids1,attn_mask1,token_type_ids1,label = batch['input_ids1'], batch['attn_mask1'], batch['token_type_ids1'],batch['label']
            if torch.cuda.is_available():
                input_ids1,attn_mask1,token_type_ids1,label = input_ids1.cuda(),attn_mask1.cuda(),token_type_ids1.cuda(),label.cuda()
            # optimizer.zero_grad()
            logits, pooled_output= model(input_ids1,attn_mask1,token_type_ids1)
            loss = nn.CrossEntropyLoss()
            loss=loss(logits,label)
            loss.backward()
            # # 对抗训练
            # fgm.attack()  # 在embedding上添加对抗扰动
            # logits, pooled_output = model(input_ids, attn_mask, token_type_ids)
            # # print(logits))
            # loss_fn = nn.CrossEntropyLoss()
            # loss_adv = loss_fn(logits,label)
            # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            # fgm.restore()  # 恢复embedding参数
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1
            logits = torch.max(logits.data, 1)[1].cpu()
            # print(logits)
            label = label.cpu()
            # print(label)
            if steps % GCONF.steps_show == 0:
                f1=f1_score(logits,label,average='macro')
                print('epoch:%d\t\t\tsteps:%d\t\t\tloss:%.6f\t\t\tf1_score:%.4f'%(epoch,steps,loss.item(),f1))
        dev_f1 = dev_eval(val_iter,model)
        # g.write(str(dev_f1)+'\n')
        print('dev\nf1:%.6f'%(dev_f1))
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model,GCONF.saved_model_path+'/'+'拼接+Mbert_'+str(fold)+'.pth')
            print('save best model\t\tf1:%.6f'%best_f1)

# 验证集进行评估
def dev_eval(val_iter,model):
    model.eval()
    logits_list=[]
    label_list=[]
    avg_loss=0
    for batch in val_iter:
        input_ids1, attn_mask1, token_type_ids1,  label = batch['input_ids1'], batch['attn_mask1'], batch['token_type_ids1'],  batch['label']
        if torch.cuda.is_available():
            input_ids1, attn_mask1, token_type_ids1, label = input_ids1.cuda(), attn_mask1.cuda(), token_type_ids1.cuda(), label.cuda()
        with torch.no_grad():
            logits, pooled_ = model(input_ids1, attn_mask1, token_type_ids1)
        loss = nn.CrossEntropyLoss()
        loss=loss(logits,label)
        avg_loss += loss.item()
        logits = torch.max(logits.data, 1)[1].cpu()
        label = label.cpu()
        logits_list.extend(logits)
        label_list.extend(label)
    f1 = f1_score(logits_list,label_list,average='macro')
    model.train()
    return f1


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



"""
从头往后开始跑实验！
"""
GCONF = GlobalConfig()
import os
if not os.path.exists(GCONF.saved_model_path):
  os.mkdir(GCONF.saved_model_path)

seed_everything(GCONF.seed)
all_labels = [0,1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained(GCONF.roberta_path, do_lower_case=True, add_prefix_space=True, is_split_into_words=True, truncation=True)

for fold in range(1):

    # 读取训练数据集
    train=pd.read_csv("../data/data_train_all.csv",index_col=0)
    train_ds = TweetBertDataset(pd.concat([train]), tokenizer, GCONF.max_length, is_testing=False)#（默认是上下堆叠）
    train_dl = DataLoader(train_ds, batch_size=GCONF.batch_size, shuffle=True)
    train_steps = (len(train_dl) * GCONF.epochs)

    dev=pd.read_csv("../data/data_dev_all.csv",index_col=0)
    valid_ds = TweetBertDataset(dev, tokenizer, GCONF.max_length, is_testing=False)
    valid_dl = DataLoader(valid_ds, batch_size=GCONF.batch_size * 4, shuffle=False)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TweetBertModel(GCONF.roberta_path)
    model.to(DEVICE)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 1e-3
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.
        }
    ]
    print(train_(train_dl, valid_dl, model,fold))
    model.to('cpu')
    del train_ds, valid_ds, model
    gc.collect()
    torch.cuda.empty_cache()






