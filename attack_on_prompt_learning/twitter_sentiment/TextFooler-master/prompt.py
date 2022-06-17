from turtle import forward
import numpy as np
import pandas as pd
import re
import emoji
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup,BertForMaskedLM
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from verbalizer import reparapgrasing,idx2word
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'
prompt_model=None

class MyDataSet(Data.Dataset):
    def __init__(self, sen , mask , typ ,label ):
        super(MyDataSet, self).__init__()
        self.sen = sen
        self.mask = mask
        self.typ = typ
        self.label = label

    def __len__(self):
        return self.sen.shape[0]

    def __getitem__(self, idx):
        return self.sen[idx], self.mask[idx],self.typ[idx],self.label[idx]


class Prompt_Model(nn.Module):
    def __init__(self,PLM,p_label_idx,n_label_idx):
        super( Prompt_Model, self ).__init__() 
        self.PLM=PLM
        # self.prompt="it was [MASK]."
        self.p_label_idx=p_label_idx
        self.n_label_idx=n_label_idx
    def forward(self, input_id, attention_mask, token_type_id):
        # try:
        #     mask_pos=int(np.argwhere(input_id.cpu().detach().numpy()==103))
        # except:
        #     mask_pos=3
        #     print("mask_pos by default")
        mask_pos=3
        output  = self.PLM(input_id, attention_mask, token_type_id)
        softmax=torch.nn.Softmax(dim=-1)
        pred_mask = softmax(output[:, mask_pos, :])
        positive_prob=torch.sum(pred_mask[:,self.p_label_idx],axis=-1)
        neg_prob=torch.sum(pred_mask[:,self.n_label_idx],axis=-1)
        return torch.cat((neg_prob.reshape(-1,1),positive_prob.reshape(-1,1)),-1)
        

class Bert_Model(nn.Module):
    def __init__(self,  bert_path ,config_file ):
        super(Bert_Model, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path,config=config_file)  # 加载预训练模型权重
        # self.softmax=torch.nn.Softmax(dim=-1)


    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logit = outputs[0]  # 池化后的输出 [bs, config.hidden_size]
        return logit 


def data_clean(tweet):
    r = "[_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」“”‘’（）{}【】□‡()/／\\\[\]\" ¦]"
    tweet = emoji.replace_emoji(tweet, replace=' ')                    # 删除emoji
    tweet = re.sub(r'[http|https]*://[a-zA-Z0-9.?/&=:]*', ' ', tweet)  # 删除网址
    tweet = re.sub(r'@\S+', ' ', tweet)                                # 删除 @用户 
    tweet = re.sub(r, ' ', tweet)                                      # 删除特殊符号
    tweet = re.sub(r'â', ' ', tweet) 
    tweet = re.sub(r'', ' ', tweet) 
    tweet = re.sub(r'', ' ', tweet) 
    tweet = re.sub(r'ð', ' ', tweet)  
    tweet = re.sub(r'\s+', ' ', tweet)                                 # 删除多余空格换行符                        
    # 删除 中日韩 三语
    tweet = re.sub('[\u4e00-\u9fa5\uac00-\ud7ff\u3040-\u309f\u30a0-\u30ff]', '', tweet)
    return tweet
 
def main():
    global prompt_model
    # x = torch.zeros(3)
    '''Load the data'''
    # read data from csv file
    # pd_reader=pd.read_csv("/home/wrc/attack_on_prompt_learning/twitter_sentiment/train.csv")
    pd_reader=pd.read_csv("/home/wrc/attack_on_prompt_learning/twitter_sentiment/TextFooler-master/data/sst-2/train.csv")
    tweet=pd_reader["text"] # the column with the text data
    label=pd_reader["label"] # the column with the label

    # exclude the value whose type is not str
    nan_list = [index for index,i in enumerate(tweet) if type(i)!=type(' ') ]
    tweet = tweet.drop(nan_list)
    
    # data clean & dataFrame to list
    total_num=69000
    translation_sentence= tweet.apply(data_clean).tolist()[:]
    label=label.tolist()[:]
    translation_sentence2=[]
    label2=[]
    cnt0=0
    cnt1=0
    for i in range(len(label)):
        if label[i]==0:
            if(cnt0)==total_num/2:
                continue
            cnt0+=1
            translation_sentence2.append(translation_sentence[i])
            label2.append(1)
        else:
           
            if(cnt1)==total_num-total_num/2:
                break
            cnt1+=1
            translation_sentence2.append(translation_sentence[i])
            label2.append(0)
    
    # use sklearn function to split the dataset test_size=0.3
    x_train, x_test, y_train, y_test = train_test_split(translation_sentence2, label2, test_size=0.3,random_state=42)
    
    print("测试数据共：",len(y_test), len(x_test))


    '''Tokenization'''
    ## load the tokenizer
    vocab_path = "/home/wrc/attack_on_prompt_learning/twitter_sentiment/vocab.txt"
    tokenizer = BertTokenizer.from_pretrained(vocab_path) #unknown word will be mapped to 100

    # template
    prefix = 'In my opinion, it is [MASK]. '
    # pos_id = tokenizer.convert_tokens_to_ids('good')  #2204
    # neg_id = tokenizer.convert_tokens_to_ids('bad')   #2919

    # tokenize
    Inputid = []
    label_train = []
    Segmentid = []
    Attention_mask = []

    for i in range(len(x_train)):
        # add the prompt & encode the new text
        text_ = prefix + x_train[i] 
        encode_dict = tokenizer.encode_plus(text_ , max_length=60 , padding='max_length', truncation=True) # encode仅返回input_id, encode_plus返回所有的编码信息
        
        inputid = encode_dict["input_ids"]               # input_ids:是单词在词典中的编码
        segment_id = encode_dict["token_type_ids"]   # token_type_ids:区分两个句子的编码（上句全为0，下句全为1）
        attention_mask = encode_dict["attention_mask"]       # attention_mask:指定对哪些词进行self-Attention
        
        Inputid.append(inputid)
        Segmentid.append(segment_id)
        Attention_mask.append(attention_mask)

        mask_pos=int(np.argwhere(np.array(inputid)==103))
    Inputid = np.array(Inputid)
    Segmentid = np.array(Segmentid)
    Attention_mask = np.array(Attention_mask)
    y_train = np.array(y_train)

    ''' Divide the train and dev set'''
    print("正在划分train set和dev set")
    #shuffle
    data_num=Inputid.shape[0]
    idxes = np.arange(data_num)  #idxes的第一维度，也就是数据大小
    np.random.seed(2019)   # 固定种子
    np.random.shuffle(idxes)
    a = int(data_num/5*4-1)
    
    # 划分训练集、验证集
    input_ids_train,  input_ids_valid  = Inputid[idxes[:a]], Inputid[idxes[a:]]
    input_masks_train,  input_masks_valid = Attention_mask[idxes[:a]], Attention_mask[idxes[a:]]
    input_types_train, input_types_valid = Segmentid[idxes[:a]], Segmentid[idxes[a:]]
    label_train, label_valid = y_train[idxes[:a]], y_train[idxes[a:]]
    # print(input_ids_train.shape, label_train.shape, input_ids_valid.shape, y_valid.shape)

    #测试集构建
    tInputid = []
    tLabelid = []
    t_Segmentid = []
    t_Attention_mask = []
    for i in range(len(x_test)):
        text_ =    prefix + x_test[i]
        encode_dict = tokenizer.encode_plus(text_ , max_length=60 , padding='max_length', truncation=True)
        
        inputid = encode_dict["input_ids"]
        segment_id = encode_dict["token_type_ids"]
        attention_mask = encode_dict["attention_mask"]
        
        tInputid.append(inputid)
        t_Segmentid.append(segment_id)
        t_Attention_mask.append(attention_mask)
    tInputid = np.array(tInputid)
    label_test = np.array(y_test)
    t_Segmentid = np.array(t_Segmentid)
    t_Attention_mask = np.array(t_Attention_mask)

    print("测试集大小",tInputid.shape , label_test.shape)
    
    
    '''Create the dataloader'''
    input_ids_train = torch.from_numpy(input_ids_train).long()
    input_ids_valid = torch.from_numpy(input_ids_valid).long()
    input_ids_test = torch.from_numpy(tInputid).long()

    input_masks_train = torch.from_numpy(input_masks_train).long()
    input_masks_valid = torch.from_numpy(input_masks_valid).long()
    input_masks_test = torch.from_numpy(t_Attention_mask).long()

    input_types_train = torch.from_numpy(input_types_train).long()
    input_types_valid = torch.from_numpy(input_types_valid).long()
    input_types_test = torch.from_numpy(t_Segmentid).long()

    label_train = torch.from_numpy(label_train).long()
    label_valid = torch.from_numpy(label_valid).long()
    label_test = torch.from_numpy(label_test).long()
    
    train_dataset = Data.DataLoader(MyDataSet(input_ids_train,  input_masks_train , input_types_train , label_train), 32, True)
    # valid_dataset = Data.DataLoader(MyDataSet(input_ids_valid,  input_masks_valid , input_types_valid , label_valid), 32, True)
    test_dataset = Data.DataLoader(MyDataSet(input_ids_test,  input_masks_test , input_types_test , label_test), 128, True)


    '''Load the model'''
    config_path = "/home/wrc/attack_on_prompt_learning/twitter_sentiment/config.json"
    config = BertConfig.from_pretrained(config_path)  # 导入模型超参数
    print(config)
    DEVICE = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
    print(DEVICE)

    print("正在加载模型")
    model = Bert_Model( bert_path="/home/wrc/attack_on_prompt_learning/twitter_sentiment/bert-base-uncased-pytorch_model.bin.1", config_file=config).to(DEVICE)
    print("模型加载完毕")
    print("正在训练中。。。")
    correct = 0

    p_seed_label={}
    n_seed_label={}
    avg_p_vec=np.zeros(30522)
    avg_n_vec=np.zeros(30522)
    label_hist=[n_seed_label,p_seed_label] #the list of dict pointers
    for idx, (ids, att, tpe, y) in enumerate(train_dataset):
        ids, att, tpe, y = ids.to(DEVICE), att.to(DEVICE), tpe.to(DEVICE), y.to(DEVICE)
        #ids, attention mask, type, y
        output  = model(ids, att, tpe)
        softmax=torch.nn.Softmax(dim=-1)
        pred_mask = softmax(output[:, mask_pos, :]).cpu().detach().numpy()
        y=y.cpu().detach().numpy()
        torch.cuda.empty_cache()
        avg_p_vec+=np.sum(pred_mask*(y.reshape(-1,1)),axis=-2)
        avg_n_vec+=np.sum(pred_mask*(1-y.reshape(-1,1)),axis=-2)
    topk=20
    avg_p_vec,avg_n_vec=avg_p_vec-avg_n_vec,avg_n_vec-avg_p_vec
    n_seed_label= np.argsort(avg_n_vec)[-topk:]
    p_seed_label= np.argsort(avg_p_vec)[-topk:]
    a=[idx2word[i] for i in n_seed_label]
    b=[idx2word[i] for i in p_seed_label]
    n_label_idx, p_label_idx=n_seed_label,p_seed_label
    # n_label_idx, p_label_idx=reparapgrasing(n_seed_label,p_seed_label)
    correct = 0
    prompt_model=Prompt_Model(model,p_label_idx,n_label_idx)
    for idx, (ids, att, tpe, y) in enumerate(test_dataset):
        ids, att, tpe, y = ids.to(DEVICE), att.to(DEVICE), tpe.to(DEVICE), y.to(DEVICE)
        output  = model(ids, att, tpe)
        softmax=torch.nn.Softmax(dim=-1)
        pred_mask = softmax(output[:, mask_pos, :])
        # pred_idx = torch.max(pred_mask.data, 1)[1]
        positive_prob=torch.sum(pred_mask[:,p_label_idx],axis=-1)
        neg_prob=torch.sum(pred_mask[:,n_label_idx],axis=-1)
        predicted=positive_prob>neg_prob
        correct += (predicted == y).sum()
    acc = float(correct / len(label_test))
    print(acc)
    print("wrc")   

main()