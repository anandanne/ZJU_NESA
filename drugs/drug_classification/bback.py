import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from block1D import ClassNet
import myutils
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'
max_epoch = 100
data_root = "/home/wrc/zhijiang/drug_classification/data" 
batch_size = 100
batch_size2 = 100
batch_size3=100
num_workers = 0
save_loc = "/home/wrc/zhijiang/drug_classification/result/checkpoint_con" 
n_features = 15
n_classes = 8
lr=0.001
length_list=[636,4614,1526,1662,4785,4989,1803]


import ImageSequence_1D as IS
shuffle1=True
shuffle2=True
shuffle3=False
def convert(x,value,y):
    index = torch.argwhere( y==value )
    pred=x[index]
    s=torch.sum(pred==value)
    return s
def find_dataset(value,x,label,con):
    index=torch.argwhere(label==value)
    new_x=x[index]
    new_x=np.squeeze(new_x)
    new_con=con[index]
    new_con=np.squeeze(new_con)
    return new_x,new_con

#可改训练测试数据数量！！！
train_loader =IS.get_loader(data_root, batch_size, shuffle1, num_workers,0)#0 refers to train
eval_loader = IS.get_loader(data_root, batch_size2, shuffle2, num_workers,1)#1 refers to eval
# test_loader=IS.get_loader(data_root, batch_size3, shuffle3, num_workers,2)
# x=test_loader
test_dataset=IS.DatasetSequence(data_root,2)
features=test_dataset.data_input
drug_label=test_dataset.label
con_label=test_dataset.label2
test0,con_label0=find_dataset(0,features,drug_label,con_label)
test1,con_label1=find_dataset(1,features,drug_label,con_label)
test2,con_label2=find_dataset(2,features,drug_label,con_label)
test3,con_label3=find_dataset(3,features,drug_label,con_label)
test4,con_label4=find_dataset(4,features,drug_label,con_label)
test5,con_label5=find_dataset(5,features,drug_label,con_label)
test6,con_label6=find_dataset(6,features,drug_label,con_label)

# c1=torch.argwhere(con_label1==0.08)
# test1_1=test1[c1].cuda(1)
# c2=torch.argwhere(con_label1==0.016)
# test1_2=test1[c2].cuda(1)
# c3=torch.argwhere(con_label1==0.4)
# test1_3=test1[c3].cuda(1)
# c4=torch.argwhere(con_label1==2)
# test1_4=test1[c4].cuda(1)

# test_loader = IS.get_loader(data_root, batch_size3, shuffle3, num_workers,2)#1 refers to eval

model = ClassNet(n_features, n_classes)
model=model.cuda(1)
writer = SummaryWriter()
optimizer = torch.optim.Adam(model.parameters(), lr)
# out1_1=model(test1_1,n_features)
# out1_1=out1_1.cpu().detach().numpy()
# out1_1=out1_1.squeeze(1) 
# # std_mean1_1=torch.std_mean(out1_1)
# out1_2=model(test1_2,n_features)
# out1_2=out1_2.cpu().detach().numpy()
# out1_2=out1_2.squeeze(1) 
# # std_mean1_2=torch.std_mean(out1_2)
# out1_3=model(test1_3,n_features)
# out1_3=out1_3.cpu().detach().numpy() 
# out1_3=out1_3.squeeze(1)
# # std_mean1_3=torch.std_mean(out1_3)
# out1_4=model(test1_4,n_features)
# out1_4=out1_4.cpu().detach().numpy() 
# out1_4=out1_4.squeeze(1)
# # std_mean1_4=torch.std_mean(out1_4)
# labels1='0.016','0.08','0.4','2'
# plt.grid(True)  # 显示网格
# plt.boxplot([out1_2,out1_1,out1_3,out1_4],
#             medianprops={'color': 'red', 'linewidth': '1.5'},
#             meanline=True,
#             showmeans=True,
#             meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
#             flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
#             labels=labels1)
# plt.yticks(np.arange(0.4, 0.81, 0.1))
# plt.show()
# plt.savefig('concentration.jpg')
# loss_func = nn.CrossEntropyLoss()
loss_func2=nn.MSELoss()
loss_func=nn.CrossEntropyLoss()
best_acc=0
train_acc_epoches=[]
train_loss_epoches=[]
eval_acc_epoches=[]



for epoch in range(max_epoch):
    print('epoch {}'.format(epoch + 1))
    # training process
    train_loss = 0.
    train_acc_num = 0.
    train_acc = 0.
    lenth = 0.
    model.train()
    for i, (batch_x, batch_y,_) in enumerate(train_loader):
        batch_x = batch_x.cuda(1)
        batch_y = batch_y.cuda(1)
        # batch_con=batch_y.cuda(1)
        batch_x = batch_x.float()
        # batch_con=batch_con.float()
        out = model(batch_x,n_features)
        out1=out[:,:7]
        #-===========================================================
        # print(out, out.type)
        # print(batch_y, batch_y.type)
        
        # # pred = torch.max(out, 1)[1].type(torch.FloatTensor)
        # # print(pred, pred.shape)
        # break
        
        loss = loss_func(out1, batch_y)
        # loss=loss.float()
        train_loss += loss.item()
        pred = torch.max(out1, 1)[1]                             #？？
        train_correct = (pred == batch_y).sum()
        
        # print('i:',i)
        # print('train_correct.item:',train_correct.item())
        # print('-------------')
        train_acc_num += train_correct.item()
        train_acc = train_acc_num/((i+1)*batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lenth = i
    
    train_loss = train_loss/((lenth+1))
    writer.add_scalar("loss_train", train_loss, epoch)
    writer.add_scalar("acc_train", train_acc, epoch)
    train_acc_epoches.append(train_acc)
    train_loss_epoches.append(train_loss)
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss, train_acc ))

    model.eval()
    eval_loss = 0.
    eval_acc_num = 0.
    eval_acc = 0.
    lenth = .0
    is_last = False
    # eval process
    for j,(batch_x, batch_y,_) in enumerate(eval_loader):
        batch_x, batch_y = batch_x.cuda(1), batch_y.cuda(1)
        batch_x = batch_x.float()
        out = model(batch_x,n_features)
        out1=out[:,:7]
        loss = loss_func(out1, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out1, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc_num += num_correct.item()
        eval_acc = eval_acc_num/((j+1)*batch_size2)
        lenth = j
    if(eval_acc>best_acc):
        is_last=True
        best_acc=eval_acc    
    eval_loss = eval_loss/((lenth+1))
    # eval_acc_epoches.append(eval_acc)
    writer.add_scalar("loss_eval", eval_loss, epoch)
    writer.add_scalar("acc_eval", eval_acc, epoch)
    print('Validation Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss , eval_acc))

    # save checkpoint as well as last one as "model_best.pth"
    
    myutils.save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'last_test_loss': eval_loss,
            'lr': optimizer.param_groups[-1]['lr']
        }, save_loc, is_last, filename='checkpoint_{}.pth'.format(epoch))

writer.close()

for epoch in range(max_epoch):
    print('epoch {}'.format(epoch + 1))
    # training process
    train_loss = 0.
    train_acc_num = 0.
    train_acc = 0.
    lenth = 0.
    model.train()
    for i, (batch_x, _,batch_con) in enumerate(train_loader):
        batch_x = batch_x.cuda(1)
        batch_con=batch_con.cuda(1)
        batch_x = batch_x.float()
        batch_con=batch_con.float()
        out = model(batch_x,n_features)
        out1=out[:,7]
        #-===========================================================
        # print(out, out.type)
        # print(batch_y, batch_y.type)
        
        # # pred = torch.max(out, 1)[1].type(torch.FloatTensor)
        # # print(pred, pred.shape)
        # break
        
        loss = loss_func2(out1, batch_con)
        # loss=loss.float()
        train_loss += loss.item()
        # pred = torch.max(out1, 1)[1]                             #？？
        # train_correct = (pred == batch_y).sum()
        
        # print('i:',i)
        # print('train_correct.item:',train_correct.item())
        # print('-------------')
        # train_acc_num += train_correct.item()
        # train_acc = train_acc_num/((i+1)*batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lenth = i
    
    train_loss = train_loss/((lenth+1))
    writer.add_scalar("loss_train", train_loss, epoch)
    # writer.add_scalar("acc_train", train_acc, epoch)
    train_acc_epoches.append(train_acc)
    train_loss_epoches.append(train_loss)
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss, train_acc ))

    model.eval()
    eval_loss = 0.
    eval_acc_num = 0.
    eval_acc = 0.
    lenth = .0
    is_last = False
    # eval process
    for j,(batch_x, _,batch_con) in enumerate(eval_loader):
        batch_x, batch_con = batch_x.cuda(1), batch_con.cuda(1)
        batch_x = batch_x.float()
        out = model(batch_x,n_features)
        out1=out[:,7]
        loss = loss_func2(out1, batch_con)
        eval_loss += loss.item()
        # pred = torch.max(out1, 1)[1]
        # num_correct = (pred == batch_y).sum()
        # eval_acc_num += num_correct.item()
        # eval_acc = eval_acc_num/((j+1)*batch_size2)
        lenth = j
    # if(eval_acc>best_acc):
    #     is_last=True
    #     best_acc=eval_acc    
    eval_loss = eval_loss/((lenth+1))
    # eval_acc_epoches.append(eval_acc)
    writer.add_scalar("loss_eval", eval_loss, epoch)
    # writer.add_scalar("acc_eval", eval_acc, epoch)
    print('Validation Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss , eval_acc))

    # save checkpoint as well as last one as "model_best.pth"
    
    myutils.save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'last_test_loss': eval_loss,
            'lr': optimizer.param_groups[-1]['lr']
        }, save_loc, is_last, filename='checkpoint_con_{}.pth'.format(epoch))

writer.close()
#     # save checkpoint as well as last one as "model_best.pth"
#     if_last=False
#     if epoch==max_epoch-1:
#         is_last=True
#     myutils.save_checkpoint({
#         'epoch': epoch,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'last_test_loss': eval_loss,
#             'lr': optimizer.param_groups[-1]['lr']
#         }, save_loc, is_last, filename='checkpoint_{}.pth'.format(epoch))

# writer.close()
# #test part
#drug0
test0=test0.unsqueeze(1).cuda(1)
out0=model(test0,n_features)
std_mean0=torch.std_mean(out0)
#drug1
c1=torch.argwhere(con_label1==0.08)
test1_1=test1[c1].cuda(1)
c2=torch.argwhere(con_label1==0.016)
test1_2=test1[c2].cuda(1)
c3=torch.argwhere(con_label1==0.4)
test1_3=test1[c3].cuda(1)
c4=torch.argwhere(con_label1==2)
test1_4=test1[c4].cuda(1)
#drug2
c1=torch.argwhere(con_label2==0.08)
test2_1=test2[c1].cuda(1)
c2=torch.argwhere(con_label2==0.016)
test2_2=test2[c2].cuda(1)
c3=torch.argwhere(con_label2==0.4)
test2_3=test2[c3].cuda(1)
c4=torch.argwhere(con_label2==2)
test2_4=test2[c4].cuda(1)
#drug3
c1=torch.argwhere(con_label3==0.08)
test3_1=test3[c1].cuda(1)
c2=torch.argwhere(con_label3==0.016)
test3_2=test3[c2].cuda(1)
c3=torch.argwhere(con_label3==0.4)
test3_3=test3[c3].cuda(1)
c4=torch.argwhere(con_label3==2)
test3_4=test3[c4].cuda(1)
#drug4
c1=torch.argwhere(con_label4==0.11)
test4_1=test4[c1].cuda(1)
c2=torch.argwhere(con_label4==0.54)
test4_2=test4[c2].cuda(1)
c3=torch.argwhere(con_label4==0.0216)
test4_3=test4[c3].cuda(1)
c4=torch.argwhere(con_label4==0.00432)
test4_4=test4[c4].cuda(1)
#drug5
c1=torch.argwhere(con_label5==0.08)
test5_1=test5[c1].cuda(1)
c2=torch.argwhere(con_label5==0.016)
test5_2=test5[c2].cuda(1)
c3=torch.argwhere(con_label5==0.4)
test5_3=test5[c3].cuda(1)
c4=torch.argwhere(con_label5==2)
test5_4=test5[c4].cuda(1)
#drug6
c1=torch.argwhere(con_label6==0.08)
test6_1=test6[c1].cuda(1)
c2=torch.argwhere(con_label6==0.016)
test6_2=test6[c2].cuda(1)
c3=torch.argwhere(con_label6==0.4)
test6_3=test6[c3].cuda(1)
c4=torch.argwhere(con_label6==2)
test6_4=test6[c4].cuda(1)
#输出
out1_1=model(test1_1,n_features)
out1_1=out1_1[:,7]
out1_1=out1_1.cpu().detach().numpy()
# out1_1=out1_1.squeeze(1) 

out1_2=model(test1_2,n_features)
out1_2=out1_2[:,7]
out1_2=out1_2.cpu().detach().numpy()
# out1_2=out1_2.squeeze(1) 

out1_3=model(test1_3,n_features)
out1_3=out1_3[:,7]
out1_3=out1_3.cpu().detach().numpy()
# out1_3=out1_3.squeeze(1) 

out1_4=model(test1_4,n_features)
out1_4=out1_4[:,7]
out1_4=out1_4.cpu().detach().numpy()
# out1_4=out1_4.squeeze(1) 
labels1='0.016','0.08','0.4','2'
labels2='0.00432','0.0216','0.11','0.54'
plt.figure()
plt.grid(True)  # 显示网格
plt.boxplot([out1_2,out1_1,out1_3,out1_4],0,'',
            medianprops={'color': 'red', 'linewidth': '1.5'},
            meanline=True,
            showmeans=True,
            vert=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
            labels=labels1)
plt.yticks(np.arange(0.4, 0.81, 0.1))
plt.show()
plt.savefig('concentration1.jpg')

out2_1=model(test2_1,n_features)
out2_1=out2_1[:,7]
out2_1=out2_1.cpu().detach().numpy()
# out2_1=out2_1.squeeze(1) 

out2_2=model(test2_2,n_features)
out2_2=out2_2[:,7]
out2_2=out2_2.cpu().detach().numpy()
# out2_2=out2_2.squeeze(1) 

out2_3=model(test2_3,n_features)
out2_3=out2_3[:,7]
out2_3=out2_3.cpu().detach().numpy()
# out2_3=out2_3.squeeze(1) 

out2_4=model(test2_4,n_features)
out2_4=out2_4[:,7]
out2_4=out2_4.cpu().detach().numpy()
# out2_4=out2_4.squeeze(1) 
plt.figure()
plt.grid(True)  # 显示网格
plt.boxplot([out2_2,out2_1,out2_3,out2_4],0,'',
            medianprops={'color': 'red', 'linewidth': '1.5'},
            meanline=True,
            showmeans=True,
            vert=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
            labels=labels1)
plt.yticks(np.arange(0.4, 0.81, 0.1))
plt.show()
plt.savefig('concentration2.jpg')

out3_1=model(test3_1,n_features)
out3_1=out3_1[:,7]
out3_1=out3_1.cpu().detach().numpy()
# out3_1=out3_1.squeeze(1) 

out3_2=model(test3_2,n_features)
out3_2=out3_2[:,7]
out3_2=out3_2.cpu().detach().numpy()
# out3_2=out3_2.squeeze(1) 

out3_3=model(test3_3,n_features)
out3_3=out3_3[:,7]
out3_3=out3_3.cpu().detach().numpy()
# out3_3=out3_3.squeeze(1) 

out3_4=model(test3_4,n_features)
out3_4=out3_4[:,7]
out3_4=out3_4.cpu().detach().numpy()
# out3_4=out3_4.squeeze(1) 
plt.figure()
plt.grid(True)  # 显示网格
plt.boxplot([out3_2,out3_1,out3_3,out3_4],0,'',
            medianprops={'color': 'red', 'linewidth': '1.5'},
            meanline=True,
            showmeans=True,
            vert=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
            labels=labels1)
plt.yticks(np.arange(0.4, 0.81, 0.1))
plt.show()
plt.savefig('concentration3.jpg')

out4_1=model(test4_1,n_features)
out4_1=out4_1[:,7]
out4_1=out4_1.cpu().detach().numpy()
# out4_1=out4_1.squeeze(1) 

out4_2=model(test4_2,n_features)
out4_2=out4_2[:,7]
out4_2=out4_2.cpu().detach().numpy()
# out4_2=out4_2.squeeze(1) 

out4_3=model(test4_3,n_features)
out4_3=out4_3[:,7]
out4_3=out4_3.cpu().detach().numpy()
# out4_3=out4_3.squeeze(1) 

out4_4=model(test4_4,n_features)
out4_4=out4_4[:,7]
out4_4=out4_4.cpu().detach().numpy()
# out4_4=out4_4.squeeze(1) 
plt.figure()
plt.grid(True)  # 显示网格
plt.boxplot([out4_4,out4_3,out4_1,out4_2],0,'',
            medianprops={'color': 'red', 'linewidth': '1.5'},
            meanline=True,
            showmeans=True,
            vert=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
            labels=labels2)
plt.yticks(np.arange(0.4, 0.81, 0.1))
plt.show()
plt.savefig('concentration4.jpg')
out5_1=model(test5_1,n_features)
out5_1=out5_1[:,7]
out5_1=out5_1.cpu().detach().numpy()
# out5_1=out5_1.squeeze(1)

out5_2=model(test5_2,n_features)
out5_2=out5_2[:,7]
out5_2=out5_2.cpu().detach().numpy()
# out5_2=out5_2.squeeze(1) 

out5_3=model(test5_3,n_features)
out5_3=out5_3[:,7]
out5_3=out5_3.cpu().detach().numpy()
# out5_3=out5_3.squeeze(1) 

out5_4=model(test5_4,n_features)
out5_4=out5_4[:,7]
out5_4=out5_4.cpu().detach().numpy()
# out5_4=out5_4.squeeze(1) 
plt.figure()
plt.grid(True)  # 显示网格
plt.boxplot([out5_2,out5_1,out5_3,out5_4],0,'',
            medianprops={'color': 'red', 'linewidth': '1.5'},
            meanline=True,
            showmeans=True,
            vert=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
            labels=labels1)
plt.yticks(np.arange(0.4, 0.81, 0.1))
plt.show()
plt.savefig('concentration5.jpg')
out6_1=model(test6_1,n_features)
out6_1=out6_1[:,7]
out6_1=out6_1.cpu().detach().numpy()
# out6_1=out6_1.squeeze(1) 

out6_2=model(test6_2,n_features)
out6_2=out6_2[:,7]
out6_2=out6_2.cpu().detach().numpy()
# out6_2=out6_2.squeeze(1)

out6_3=model(test6_3,n_features)
out6_3=out6_3[:,7]
out6_3=out6_3.cpu().detach().numpy()
# out6_3=out6_3.squeeze(1) 

out6_4=model(test6_4,n_features)
out6_4=out6_4[:,7]
out6_4=out6_4.cpu().detach().numpy()
# out6_4=out6_4.squeeze(1) 
plt.figure()
plt.grid(True)  # 显示网格
plt.boxplot([out6_2,out6_1,out6_3,out6_4],0,'',
            medianprops={'color': 'red', 'linewidth': '1.5'},
            meanline=True,
            showmeans=True,
            vert=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
            labels=labels1)
plt.yticks(np.arange(0.4, 0.81, 0.1))
plt.show()
plt.savefig('concentration6.jpg')



# # epoches=np.arange(1,101,1)
# # plt.plot(epoches,eval_acc_epoches0,label="DMSO")
# # plt.plot(epoches,eval_acc_epoches1,label="acetylcholine")
# # plt.plot(epoches,eval_acc_epoches2,label="astemizole")
# # plt.plot(epoches,eval_acc_epoches3,label="cisapride")
# # plt.plot(epoches,eval_acc_epoches4,label="endothelin")
# # plt.plot(epoches,eval_acc_epoches5,label="norepinephrine")
# # plt.plot(epoches,eval_acc_epoches6,label="sertindole")
# # plt.legend(loc ="lower right")
# # plt.xlabel("epoch")
# # plt.ylabel("accuracy")

# # plt.title('sensitivity of drug classification vs epoch')

# # fig=plt.figure()
# # loss_fig=fig.add_subplot(111)
# # lg1=loss_fig.plot(epoches,train_loss_epoches,'r-o',label="Train loss")
# # lg2=loss_fig.plot(epoches,eval_loss_epoches,'b-o',label="Evaluation loss")
# # loss_fig.set_xlabel("Epoch")
# # loss_fig.set_ylabel("Loss")
# # acc_fig = loss_fig.twinx() 
# # lg3=acc_fig.plot(epoches,train_acc_epoches,'g-D',label="Train accuracy")
# # lg4=acc_fig.plot(epoches,eval_acc_epoches,'y-D',label="Evaluation accuracy")
# # acc_fig.set_ylabel("Accuracy")

# # lg = lg1 + lg2 + lg3 +lg4
# # labels = [l.get_label() for l in lg]
# # loss_fig.legend(lg, labels, loc=7)


