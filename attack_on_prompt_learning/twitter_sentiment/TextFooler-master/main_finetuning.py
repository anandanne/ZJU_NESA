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

class Bert_Model(nn.Module):
    def __init__(self,  bert_path ,config_file ):
        super(Bert_Model, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path,config=config_file)  # 加载预训练模型权重
        # self.softmax=torch.nn.Softmax(dim=-1)
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logit = outputs[0]  # 池化后的输出 [bs, config.hidden_size]
        return logit 

class NeuralNet(nn.Module):
    def __init__(self,in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @param l(x,y) an () tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 2 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        
        super(NeuralNet, self).__init__()
        self.in_size=in_size
        self.out_size=out_size
        self.layer=torch.nn.Sequential(
            torch.nn.Linear(self.in_size,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,self.out_size)
        )

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        #torch.ones(x.shape[0], 1)
        y=self.layer(x)
        return y




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

    '''Load the data'''
    # read data from csv file
    pd_reader=pd.read_csv("/home/wrc/attack_on_prompt_learning/twitter_sentiment/train.csv")
    tweet=pd_reader["tweet"] # the column with the text data
    label=pd_reader["label"] # the column with the label
    # exclude the value whose type is not str
    nan_list = [index for index,i in enumerate(tweet) if type(i)!=type(' ') ]
    tweet = tweet.drop(nan_list)
    
    # data clean & dataFrame to list
    # total_num=2000
    translation_sentence= tweet.apply(data_clean).tolist()[:]
    label=label.tolist()[:]
    translation_sentence2=[]
    label2=[]
    cnt0=0
    cnt1=0
    for i in range(len(label)):
        if label[i]==0:
            if(cnt0)==2240:
                continue;
            cnt0+=1
            translation_sentence2.append(translation_sentence[i])
            label2.append(0)
        else:
           
            if(cnt1)==2240:
                break;
            cnt1+=1
            translation_sentence2.append(translation_sentence[i])
            label2.append(1)



    
    # use sklearn function to split the dataset test_size=0.3
    x_train, x_test, y_train, y_test = train_test_split(translation_sentence2, label2, test_size=0.3,random_state=42)
    
    print("测试数据共：",len(y_test), len(x_test))


    '''Tokenization'''
    ## load the tokenizer
    vocab_path = "/home/wrc/attack_on_prompt_learning/twitter_sentiment/vocab.txt"
    tokenizer = BertTokenizer.from_pretrained(vocab_path) #unknown word will be mapped to 100

    # template
    # prefix = '[CLS]'
    # pos_id = tokenizer.convert_tokens_to_ids('good')  #2204
    # neg_id = tokenizer.convert_tokens_to_ids('bad')   #2919

    # tokenize
    Inputid = []
    label_train = []
    Segmentid = []
    Attention_mask = []

    for i in range(len(x_train)):
        # add the prompt & encode the new text
        text_ = x_train[i] 
        encode_dict = tokenizer.encode_plus(text_ , max_length=60 , padding='max_length', truncation=True) # encode仅返回input_id, encode_plus返回所有的编码信息
        
        inputid = encode_dict["input_ids"]               # input_ids:是单词在词典中的编码
        segment_id = encode_dict["token_type_ids"]   # token_type_ids:区分两个句子的编码（上句全为0，下句全为1）
        attention_mask = encode_dict["attention_mask"]       # attention_mask:指定对哪些词进行self-Attention
        
        Inputid.append(inputid)
        Segmentid.append(segment_id)
        Attention_mask.append(attention_mask)

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
    a = int(data_num/3*2-1)
    
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
        text_ =x_test[i]
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
    model2=NeuralNet(30522,2).to(DEVICE)
    print("模型加载完毕")
    print("正在训练中。。。")
    optimizer = AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4)  #使用Adam优化器
    optimizer2 = AdamW(model2.parameters(),lr=2e-5,weight_decay=1e-4)  #使用Adam优化器
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    EPOCH = 20
    schedule = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=len(train_dataset),num_training_steps=EPOCH*len(train_dataset))
    print("正在训练中。。。")
    for epoch in range(EPOCH):

        correct = 0
        train_loss_sum = 0.0
        model.train()
        print("***** Running training epoch {} *****".format(epoch + 1))

        for idx, (ids, att, tpe, y) in enumerate(train_dataset):
            ids, att, tpe, y = ids.to(DEVICE), att.to(DEVICE), tpe.to(DEVICE), y.to(DEVICE)
            out_train  = model(ids, att, tpe)
            out_train= out_train[:, 0, :]
            out_train2=model2.forward(out_train)
            # print(out_train2.view(-1, 2).shape, y.view(-1).shape)
            loss = loss_func(out_train2.view(-1, 2), y.view(-1))
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()
            schedule.step()
            train_loss_sum += loss.item()

            if (idx + 1) % 10000 == 0:
                print("Epoch {:04d} | Step {:06d}/{:06d} | Loss {:.4f} ".format(
                    epoch + 1, idx + 1, len(train_dataset), train_loss_sum / (idx + 1)))

            pred_train=torch.argmax(out_train2, dim=1)
             
            # out_train_mask = out_train[:, 3, :]

            # predicted = torch.max(out_train_mask.data, 1)[1]
            correct += (pred_train == y).sum()
            correct = np.float(correct)
        acc = float(correct / len(label_train))
        print("epoch",epoch+1,"train accuracy: ",acc)
        eval_loss_sum = 0.0
        model.eval()
        correct = 0

    for ids, att, tpe, y in test_dataset:
        ids, att, tpe, y = ids.to(DEVICE), att.to(DEVICE), tpe.to(DEVICE), y.to(DEVICE)
        out_test = model(ids , att , tpe)
        out_test= out_test[:, 0, :]
        out_test2=model2.forward(out_test)
        loss_eval = loss_func(out_test2.view(-1, 2), y.view(-1))
        eval_loss_sum += loss_eval.item()
        # ttruelabel = y[:, 3]
        # tout_train_mask = out_test[:, 3, :]
        # predicted_test = torch.max(tout_train_mask.data, 1)[1]
        # correct_test += (predicted_test == ttruelabel).sum()
        pred_test=torch.argmax(out_test2, dim=1)
        
        # out_train_mask = out_train[:, 3, :]

        # predicted = torch.max(out_train_mask.data, 1)[1]
        correct += (pred_test == y).sum()
        correct = np.float(correct)
    acc = float(correct / len(label_test))
    print("epoch",epoch+1,"test accuracy: ",acc)
    # topk=6
    # n_seed_label= torch.argsort(avg_n_vec)[-topk:]
    # p_seed_label= torch.argsort(avg_p_vec)[-topk:]
    # a=[idx2word[i] for i in n_seed_label]
    # b=[idx2word[i] for i in p_seed_label]
    # n_label_idx, p_label_idx=reparapgrasing(n_seed_label,p_seed_label)
    # correct = 0
    # for idx, (ids, att, tpe, y) in enumerate(test_dataset):
    #     ids, att, tpe, y = ids.to(DEVICE), att.to(DEVICE), tpe.to(DEVICE), y.to(DEVICE)
    #     output  = model(ids, att, tpe)
    #     softmax=torch.nn.Softmax(dim=-1)
    #     pred_mask = softmax(output[:, 3, :])
    #     pred_idx = torch.max(pred_mask.data, 1)[1]
    #     positive_prob=torch.sum(pred_mask[:,p_label_idx],axis=-1)
    #     neg_prob=torch.sum(pred_mask[:,n_label_idx],axis=-1)
    #     predicted=positive_prob>neg_prob
        
    #     correct += (predicted == y).sum()
    # acc = float(correct / len(label_test))
    # print(acc)
    print("wrc")   


if __name__=="__main__":
    main()