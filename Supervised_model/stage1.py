import random
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, BertTokenizerFast, get_linear_schedule_with_warmup
import pandas
import os
import json
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')

"""
bert-base-multilingual-cased
malay-huggingface/bert-base-bahasa-cased
"""

# 设置随机种子
def seed_everything(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 进行全局参数的设置
# @title GlobalConfig
class GlobalConfig:
    def __init__(self):
        self.seed = 2022
        self.path = Path('./data/')
        self.max_length = 128

        self.roberta_path = 'models_malay'  # @param
        self.num_workers = os.cpu_count()
        """
        Batch Size的大小影响模型的优化程度和速度。同时其直接影响到GPU内存的使用情况，假如GPU内存不大，该数值最好设置小一点。
        """
        self.batch_size = 4  # 一次epochs更新4次模型 一次训练所选取的样本数

        self.steps_show = 100
        self.accum_steps = 1

        num_epochs = 10  # @param
        self.epochs = num_epochs
        self.warmup_steps = 0
        lr = 5e-6  # @param
        self.lr = lr  # modified from 3e-5
        run_id = "stage1_2data"  # @param
        self.offline = True
        self.saved_model_path = run_id
        self.n_splits = 5


# 将数据添加到CPU或GPU里面
def move_to_device(x, device):
    if callable(getattr(x, 'to', None)): return x.to(device)
    if isinstance(x, (tuple, list)):
        return [move_to_device(o, device) for o in x]
    elif isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


# 数据集合的构建
class TweetBertDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_testing = is_testing

    def __len__(self):
        return self.df.shape[0]

    # Token embedding，Segment embedding，Position embedding
    def __getitem__(self, ix):
        # 我们需要

        Sen = self.df.iloc[ix]['Sentence']
        Def = self.df.iloc[ix]['Definition']
        # 对得到的query和answer，进行编码工作！
        input_ids_Sen = tokenizer.encode(Sen)
        input_ids_Def = tokenizer.encode(Def)
        # 就类似那个102是sep的位置编码，需要对应修改成你用的模型的sep位置编码
        input_ids1 = input_ids_Sen
        input_ids1 = input_ids1[:-1]
        # 一句话最大长度，512位置编码 Token embedding;对应上文的input_id，用tokenizer转化文本获得。所有任务都需要有这个输入
        input_ids1 = input_ids1[:self.max_length - 1] + [102]

        input_ids2 = input_ids_Def
        input_ids2 = input_ids2[:-1]
        input_ids2 = input_ids2[1:self.max_length - 1] + [102]

        #Positional Encoding
        attn_mask1 = [1] * len(input_ids1)  #位置编码对应attention_mask。用来界定注意力机制范围1表示BERT关注该位置，0表示[PAD]不用关注
        attn_mask2 = [1] * len(input_ids2)

        """
            Segment embedding：在NSP任务中，用于区分第一句和第二句。segment embedding中只有 0 和 1两个值，
            第一句所有的token（包括cls和紧随第一句的sep）的segment embedding的值为0，
            第二句所有的token（包括紧随第二句的sep）的segment embdding的值为1。
            对应上文的token_type_ids，用来识别句子界限，如果是单句子分类任务，默认全为0，可以不用设置该参数；
            如果是文本蕴含关系等句对任务，则第一个句子对应的位置全0，句子间分隔符[SEP]处置为0，第二个句子对应位置全为1，
            句尾分隔符处置为1，[PAD]处全置0
            因为我们是单个句子，所以均为0
        """
        token_type_ids1 = [0] * len(input_ids1)  # 该部分参数在训练时不参与更新。Segment embedding而文本的位置信息又非常重要，如果是单句子分类任务，默认全为0，
        token_type_ids2 = [0] * len(input_ids2)  # 我们默认的就是一句话中含有缩略语，然后根据上下文去找答案

        # 三个编码未满的都填充0
        pad_len1 = self.max_length - len(input_ids1)
        pad_len2 = self.max_length - len(input_ids2)

        input_ids1 += [0] * pad_len1
        input_ids2 += [0] * pad_len2

        attn_mask1 += [0] * pad_len1
        attn_mask2 += [0] * pad_len2

        token_type_ids1 += [0] * pad_len1
        token_type_ids2 += [0] * pad_len2

        """
        torch.longtensor和torch.tensor的差别在于：

        torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型

        torch.tensor是一个类，用于生成一个单精度浮点类型的张量。
        map函数，将这个三个向量input_ids1, attn_mask1, token_type_ids1转换成torch.LongTensor
        """
        input_ids1, attn_mask1, token_type_ids1 = map(torch.LongTensor, [input_ids1, attn_mask1, token_type_ids1])
        input_ids2, attn_mask2, token_type_ids2 = map(torch.LongTensor, [input_ids2, attn_mask2, token_type_ids2])

        encoded_dict = {
            'input_ids1': input_ids1,
            'attn_mask1': attn_mask1,
            'token_type_ids1': token_type_ids1,
            'input_ids2': input_ids2,
            'attn_mask2': attn_mask2,
            'token_type_ids2': token_type_ids2,
        }
        # 是否正在测试
        if not self.is_testing:
            # tmp = [0] * 3
            # tmp[all_labels.index(sentiment)] = 1
            # encoded_dict['sentiment'] = torch.tensor(tmp, dtype=torch.long)
            sentiment = self.df.iloc[ix]['label']
            encoded_dict['label'] = torch.tensor(all_labels.index(sentiment), dtype=torch.long)
        return encoded_dict


class TweetBertModel(nn.Module):
    def __init__(self, roberta_path):
        super().__init__()

        roberta_config = BertConfig.from_pretrained(roberta_path)
        roberta_config.output_hidden_states = True
        # roberta_config.attention_mode = 'sliding_chunks'
        # roberta_config.gradient_checkpointing = True
        self.roberta = BertModel.from_pretrained(roberta_path, config=roberta_config)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(roberta_config.hidden_size * 2, 2)  # 拼接是因为两个向量的长度
        # 得到一个2维的概率向量后，softmax然后取第二维，第二维代表的是 正确 的概率，其实在计算评价的指标那里再softmax也可以，想一想，这个该怎么写？

        torch.nn.init.normal_(self.classifier.weight, std=0.02)  # 给tensor初始化，一般是给网络中参数weight初始化，初始化参数值符合正态分布。
        # self.soft = nn.Sigmoid()

    def forward(self, input_ids1, attn_mask1, token_type_ids1, input_ids2, attn_mask2, token_type_ids2):
        # 前馈层，注意了这里就是双塔模型了！！！很重要！

        pooled_output1 = self.roberta(
            input_ids=input_ids1,
            attention_mask=attn_mask1,
            token_type_ids=token_type_ids1
        )['last_hidden_state']
        pooled_output1 = self.dropout(pooled_output1[:, 0, :])
        pooled_output2 = self.roberta(
            input_ids=input_ids2,
            attention_mask=attn_mask2,
            token_type_ids=token_type_ids2
        )['last_hidden_state']
        pooled_output2 = self.dropout(pooled_output2[:, 0, :])
        # pooled_output = self.dropout(pooled_output)
        # 两个向量进行拼接，这里有个问题！
        pooled_output = torch.cat([pooled_output1, pooled_output2], dim=1)
        # 两个向量拼接之后，传入全连接层，再进行分类
        start_logits = self.classifier(pooled_output)
        # 返回一个全连接层的张量和两个向量拼接的张量
        return start_logits, pooled_output


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

# train_(train_dl, valid_dl, model,fold)
def train_(train_iter, val_iter, model, fold):
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.AdamW(optimizer_parameters, lr=GCONF.lr)
    scheduler = get_linear_schedule_with_warmup(  # 在预热的小学习率下，模型可以慢慢趋于稳定,等模型相对稳定后再选择预先设置的学习率进行训练,使得模型收敛速度变得更快，模型效果更佳
        optimizer,
        num_warmup_steps=int(GCONF.warmup_steps * train_steps),
        num_training_steps=train_steps
    )
    steps = 0
    best_f1 = 0
    model.train()  # 启用 Batch Normalization 和 Dropout。如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
    # fgm = FGM(model)   #model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    for epoch in range(1, GCONF.epochs + 1):
        for batch in train_iter:
            input_ids1, attn_mask1, token_type_ids1, input_ids2, attn_mask2, token_type_ids2, label = batch['input_ids1'],batch['attn_mask1'],batch['token_type_ids1'], \
                                                                                                      batch['input_ids2'],batch['attn_mask2'],batch['token_type_ids2'], \
                                                                                                      batch['label']
            if torch.cuda.is_available():
                input_ids1, attn_mask1, token_type_ids1, input_ids2, attn_mask2, token_type_ids2, label = input_ids1.cuda(), attn_mask1.cuda(), token_type_ids1.cuda(), input_ids2.cuda(), attn_mask2.cuda(), token_type_ids2.cuda(), label.cuda()
            # optimizer.zero_grad()

            logits, pooled_output = model(input_ids1, attn_mask1, token_type_ids1, input_ids2, attn_mask2,
                                          token_type_ids2)

            # 交叉熵损失函数
            loss = nn.CrossEntropyLoss()
            loss = loss(logits, label)
            # 反向传播
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
            steps += 1  # torch.max这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示），第二个值是value所在的index（也就是predicted）。dim=1表示输出所在行的最大值，若改写成dim=0则输出所在列的最大值。
            # l1=torch.softmax(logits.data,1)[:,1].cpu()
            l2=F.softmax(logits.data,1)
            # logits = torch.max(logits.data, 1)[1].cpu()
            # print(logits)
            label = label.cpu()
            # print(label) 得到一个2维的概率向量后，softmax然后取第二维，第二维代表的是 正确 的概率，其实在计算评价的指标那里再softmax也可以，想一想，这个该怎么写？
            if steps % GCONF.steps_show == 0:
                f1 = f1_score(logits, label, average='macro')
                print('epoch:%d\t\t\tsteps:%d\t\t\tloss:%.6f\t\t\tf1_score:%.4f' % (epoch, steps, loss.item(), f1))
        dev_f1 = dev_eval(val_iter, model)
        print('dev\nf1:%.6f' % (dev_f1))
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model, GCONF.saved_model_path + '/' + str(fold) + '.pth')
            print('save best model\t\tf1:%.6f' % best_f1)


# 验证集评估
def dev_eval(val_iter, model):
    model.eval()
    logits_list = []
    label_list = []
    avg_loss = 0
    for batch in val_iter:
        input_ids1, attn_mask1, token_type_ids1, input_ids2, attn_mask2, token_type_ids2, label = batch['input_ids1'], batch['attn_mask1'],batch['token_type_ids1'],batch['input_ids2'],batch['attn_mask2'],batch['token_type_ids2'],batch['label']
        if torch.cuda.is_available():
            input_ids1, attn_mask1, token_type_ids1, input_ids2, attn_mask2, token_type_ids2, label = input_ids1.cuda(), attn_mask1.cuda(), token_type_ids1.cuda(), input_ids2.cuda(), attn_mask2.cuda(), token_type_ids2.cuda(), label.cuda()
        with torch.no_grad():
            logits, pooled_ = model(input_ids1, attn_mask1, token_type_ids1, input_ids2, attn_mask2, token_type_ids2)
        loss = nn.CrossEntropyLoss()
        loss = loss(logits, label)
        avg_loss += loss.item()
        logits = torch.max(logits.data, 1)[1].cpu()
        label = label.cpu()
        logits_list.extend(logits)
        label_list.extend(label)
    f1 = f1_score(logits_list, label_list, average='macro')
    model.train()
    return f1


###############################################################################################################################################################


def get_Bert_code(df,ix,tokenizer,max_length=128,is_testing=True):
        # 我们需要

        Sen = df.iloc[ix]['Sentence']
        Def = df.iloc[ix]['Definition']
        # 对得到的query和answer，进行编码工作！
        input_ids_Sen = tokenizer.encode(Sen)
        input_ids_Def = tokenizer.encode(Def)
        # 就类似那个102是sep的位置编码，需要对应修改成你用的模型的sep位置编码
        input_ids1 = input_ids_Sen
        input_ids1 = input_ids1[:-1]
        # 一句话最大长度，512位置编码 Token embedding;对应上文的input_id，用tokenizer转化文本获得。所有任务都需要有这个输入
        input_ids1 = input_ids1[:max_length - 1] + [102]

        input_ids2 = input_ids_Def
        input_ids2 = input_ids2[:-1]
        input_ids2 = input_ids2[1:max_length - 1] + [102]

        #Positional Encoding
        attn_mask1 = [1] * len(input_ids1)  #位置编码对应attention_mask。用来界定注意力机制范围1表示BERT关注该位置，0表示[PAD]不用关注
        attn_mask2 = [1] * len(input_ids2)

        """
            Segment embedding：在NSP任务中，用于区分第一句和第二句。segment embedding中只有 0 和 1两个值，
            第一句所有的token（包括cls和紧随第一句的sep）的segment embedding的值为0，
            第二句所有的token（包括紧随第二句的sep）的segment embdding的值为1。
            对应上文的token_type_ids，用来识别句子界限，如果是单句子分类任务，默认全为0，可以不用设置该参数；
            如果是文本蕴含关系等句对任务，则第一个句子对应的位置全0，句子间分隔符[SEP]处置为0，第二个句子对应位置全为1，
            句尾分隔符处置为1，[PAD]处全置0
            因为我们是单个句子，所以均为0
        """
        token_type_ids1 = [0] * len(input_ids1)  # 该部分参数在训练时不参与更新。Segment embedding而文本的位置信息又非常重要，如果是单句子分类任务，默认全为0，
        token_type_ids2 = [0] * len(input_ids2)  # 我们默认的就是一句话中含有缩略语，然后根据上下文去找答案

        # 三个编码未满的都填充0
        pad_len1 = max_length - len(input_ids1)
        pad_len2 = max_length - len(input_ids2)

        input_ids1 += [0] * pad_len1
        input_ids2 += [0] * pad_len2

        attn_mask1 += [0] * pad_len1
        attn_mask2 += [0] * pad_len2

        token_type_ids1 += [0] * pad_len1
        token_type_ids2 += [0] * pad_len2

        """
        torch.longtensor和torch.tensor的差别在于：

        torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型

        torch.tensor是一个类，用于生成一个单精度浮点类型的张量。
        map函数，将这个三个向量input_ids1, attn_mask1, token_type_ids1转换成torch.LongTensor
        """
        input_ids1, attn_mask1, token_type_ids1 = map(torch.LongTensor, [input_ids1, attn_mask1, token_type_ids1])
        input_ids2, attn_mask2, token_type_ids2 = map(torch.LongTensor, [input_ids2, attn_mask2, token_type_ids2])

        encoded_dict = {
            'input_ids1': input_ids1,
            'attn_mask1': attn_mask1,
            'token_type_ids1': token_type_ids1,
            'input_ids2': input_ids2,
            'attn_mask2': attn_mask2,
            'token_type_ids2': token_type_ids2,
        }
        # 是否正在测试
        if not is_testing:
            # tmp = [0] * 3
            # tmp[all_labels.index(sentiment)] = 1
            # encoded_dict['sentiment'] = torch.tensor(tmp, dtype=torch.long)
            sentiment = df.iloc[ix]['label']
            encoded_dict['label'] = torch.tensor(all_labels.index(sentiment), dtype=torch.long)
        return encoded_dict











if __name__ == '__main__':

    #     # 构造全局配置实例对象
    GCONF = GlobalConfig()
    import os
    #     # 设置模型保存路径
    if not os.path.exists(GCONF.saved_model_path):
        os.mkdir(GCONF.saved_model_path)
    #     # 设置随机种子
    seed_everything(GCONF.seed)
    #     # 两种标签
    all_labels = [0,1]
    #     # 设置cpu或gpu环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     # 进行分词操作
    tokenizer = BertTokenizerFast.from_pretrained(GCONF.roberta_path, do_lower_case=True, add_prefix_space=True, is_split_into_words=True, truncation=True)

    for fold in range(5):

            #读取训练数据

            # 将数据装换成DataFrame格式
            train = pd.read_csv("../data/data_train_all.csv",index_col=0)

            print(train)
            #设置这个转换成dataframe的列索引
            # train.columns = ['ids','case1','case2','label']
            # 生成训练数据集和

            train_ds = TweetBertDataset(pd.concat([train]), tokenizer, GCONF.max_length, is_testing=False)

            train_dl = DataLoader(train_ds, batch_size=GCONF.batch_size, shuffle=True)#应该有4条数据
            # 每4个，4个一组的训练
            # for batch in train_dl:
            #     print(batch)
            #     print("#########################################################################")
            #     print(type(batch))
            #     print("#########################################################################")
            #     print(len(batch))
            #     print("#########################################################################")

            train_steps = (len(train_dl) * GCONF.epochs)#训练次数应该是10*{4*(50)}=2000  799就约等于800 800/4(batch.size)=200 一个batch选取的数据  len(train_dl)：3654

            # 读取验证集数据
            dev=pd.read_csv("../data/data_test_all.csv",index_col=0)

            valid_ds = TweetBertDataset(dev, tokenizer, GCONF.max_length, is_testing=False)
            valid_dl = DataLoader(valid_ds, batch_size=GCONF.batch_size*4, shuffle=False)#测试集每次训练13个样本
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = TweetBertModel(GCONF.roberta_path)
            model.to(DEVICE)#当我们指定了设备之后，就需要将模型加载到相应设备中，此时需要使用model=model.to(device)，将模型加载到相应的设备中。
            param_optimizer = list(model.named_parameters())#打印每一次迭代元素的名字和param。
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_parameters = [
                {
                    'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay': 1e-3 #权值衰减防止过拟合。
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

    # ds_test = pd.read_csv("../data/data_test_all.csv", index_col=0)
    # print(ds_test)
    # test_ds = TweetBertDataset(ds_test, tokenizer, GCONF.max_length, is_testing=True).__getitem__(0)
    # print(test_ds)
    # print("################################################")
    # def_re=get_Bert_code(ds_test,0,tokenizer)
    #
    # print(len(ds_test))
    # 判断tensor是否都相等---------------a.equal(b)
    # print('input_ids1',def_re['input_ids1'].equal(test_ds['input_ids1']))
    # print('attn_mask1',def_re['attn_mask1'].equal(test_ds['attn_mask1']))
    # print('token_type_ids1',def_re['token_type_ids1'].equal(test_ds['token_type_ids1']))
    # print('input_ids2',def_re['input_ids2'].equal(test_ds['input_ids2']))
    # print('attn_mask2',def_re['attn_mask2'].equal(test_ds['attn_mask2']))
    # print('token_type_ids2',def_re['token_type_ids2'].equal(test_ds['token_type_ids2']))

    # 验证结果一下

    # print(valid_ds["input_ids1"], valid_ds["attn_mask1"], valid_ds["token_type_ids1"], valid_ds["input_ids2"],
    #       valid_ds["attn_mask2"], valid_ds["token_type_ids2"])

    # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    # # 但是在实际运行中实际上需要赋值给两个变量，运行时只赋值了一个变量，我们还需要进行复试


    # logits, pooled_output = model(valid_ds["input_ids1"], valid_ds["attn_mask1"], valid_ds["token_type_ids1"],
    #                               valid_ds["input_ids2"], valid_ds["attn_mask2"], valid_ds["token_type_ids2"])

    # print("***********************************************************")
    # print(logits)
    # print("***********************************************************")
    # print(pooled_output)
    # print("***********************************************************")

    # 通过groupby句子执行
    # 先对每个样本添加正确率


    # #这个是没有标签的哈！
    # print(valid_ds)
    # # 我们需要将句子成为一个key值，然后对应不同的一句话，和多个完整形式



    # 需要传入六个参数调用模型

# matching
# save best model		f1:0.641427
# save best model		f1:0.587193
# save best model		f1:0.616390
# save best model		f1:0.659040
# save best model		f1:0.654631
# 63.17

# matching fgm
# save best model		f1:0.659401
# save best model		f1:0.618325
# save best model		f1:0.625890
# save best model		f1:0.641902
# save best model		f1:0.635034
# 63.61

# matching short
# save best model		f1:0.659032
# save best model		f1:0.597265
# save best model		f1:0.623323
# save best model		f1:0.680215
# save best model		f1:0.628917

# concat short
# save best model		f1:0.718217
# save best model		f1:0.670897
# save best model		f1:0.673351
# save best model		f1:0.678648
# save best model		f1:0.657234

# concat fgm
# save best model		f1:0.719293
# save best model		f1:0.720334
# save best model		f1:0.646516
# save best model		f1:0.732402
# save best model		f1:0.694306
