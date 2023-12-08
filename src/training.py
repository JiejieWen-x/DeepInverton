#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
from d2l import torch as d2l
from sklearn.metrics import roc_auc_score
import itertools
from sklearn.metrics import classification_report
import argparse
import os
import pandas as pd
from torch.utils import data

paser = argparse.ArgumentParser(description='training model')
paser.add_argument('-train_features', '--train_features_path', default=None, help="input the fold path of train_features")
paser.add_argument('-train_labels', '--train_labels_path', default=None, help='input the fold path of ftrain_labels')
paser.add_argument('-test_features', '--test_features_path', default=None, help="input the fold path of test_features")
paser.add_argument('-test_labels', '--test_labels_path', default=None, help='input the fold path of test_labels')
paser.add_argument('-validation_features', '--validation_features_path', default=None, help="input the fold path of validation_features")
paser.add_argument('-validation_labels', '--validaiton_labels_path', default=None, help='input the fold path of validation_labels')

paser.add_argument('-strict_features', '--strict_features_path', default=None, help="input the fold path of validation_features")
paser.add_argument('-strict_labels', '--strict_labels_path', default=None, help='input the fold path of validation_labels')

paser.add_argument('-report','--report_path',default=None,help='outpath of the report')
paser.add_argument('-m','--model_path',default=None,help='outpath of the model')
paser.add_argument('-o','--out_path',default=None,help='outpath of the model')
args = paser.parse_args()

train_features_path=args.train_features_path
train_labels_path=args.train_labels_path
test_features_path=args.test_features_path
test_labels_path=args.test_labels_path
validation_features_path=args.validation_features_path
validaiton_labels_path=args.validaiton_labels_path
strict_features_path=args.strict_features_path
strict_labels_path=args.strict_labels_path

report_path=args.report_path
model_path=args.model_path
out_path=args.out_path

def evaluate_accuracy(net, data_iter, device=None,file_name=''): 

    if isinstance(net, nn.Module):
        net.eval()  
        if not device:
            device = next(iter(net.parameters())).device

    metric = d2l.Accumulator(2)
    y_pred=[]
    y_true=[]
    probability = []
    with torch.no_grad():
        for X, y in data_iter:
           
            metric.add(d2l.accuracy(net(X), y), y.numel())
            y_hat=net(X)
            probability.append(nn.Softmax(dim=1)(y_hat).numpy().squeeze())
            y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_hat),dim=1).data
            y_pred.append((y_pred_cls.numpy()).squeeze())
            y_true.append(y.numpy().squeeze())
        
        pre=list(itertools.chain.from_iterable(y_pred))
        true=list(itertools.chain.from_iterable(y_true))
        probability=list(itertools.chain.from_iterable(probability))
        probability_df = pd.DataFrame(probability)[[1,0]].rename(columns={1:'positive', 0: 'negative'})
        auc=roc_auc_score(true, probability_df['positive'])
        predict_result = {'True_label':true,'Pre_label':pre}
        predict_result = pd.DataFrame(predict_result)
        predict_result = pd.concat([probability_df, predict_result], axis=1)
        filepath=os.path.join(report_path,file_name+'.txt')
        target_names = ['class 0', 'class 1']
        with open (filepath,'a') as file2:
            print(classification_report(true, pre, target_names=target_names),file=file2)
    return  predict_result,metric[0] / metric[1] ,auc

def train(net, train_iter, test_iter,validation_iter,strict_iter,num_epochs, lr,file_name):
    file_name=str(file_name)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
   
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            y_hat = net(X)
            
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        vali_pre_result, vali_acc, vali_auc = evaluate_accuracy(net, validation_iter)
        test_pre_result, test_acc, test_auc = evaluate_accuracy(net, test_iter)
        strict_pre_result,strict_acc,strict_auc = evaluate_accuracy(net,strict_iter)
        if vali_acc > 0.88 :
            model_path1=model_path+'/'+str(lr)+'/'+str(epoch)+'_vali_roc_'+str(test_auc)+'.pth'
            vali_pre_result_path=model_path+'/'+str(lr)+'/'+str(epoch)+'_vali_roc_'+str(vali_auc)+'.txt'
            vali_pre_result.to_csv(vali_pre_result_path,sep='\t')
            test_pre_result_path=model_path+'/'+str(lr)+'/'+str(epoch)+'_test_roc_'+str(test_auc)+'.txt'
            test_pre_result.to_csv(test_pre_result_path,sep='\t')
            strict_pre_result_path=model_path+'/'+str(lr)+'/'+str(epoch)+'_strict_roc_'+str(strict_auc)+'.txt'
            strict_pre_result.to_csv(strict_pre_result_path,sep='\t')
            torch.save(net,model_path1)
        filepath=out_path+'/'+file_name+'.txt'
        with open(filepath,'a') as file1:
            print(f' epoch {epoch}',f' lr:{lr}',f'loss {train_l:.8f}, train acc {train_acc:.8f}',f'test_acc {test_acc:.8f}',f'test_auc{test_auc:.8f}',f'vali_acc {vali_acc:.8f}',f'vali_auc {vali_auc:.8f}',f'strict_acc {strict_acc:.8f}',f'strict_auc {strict_auc:.8f}',file=file1)
    


class Reshape(torch.nn.Module):
    def forward(self,x):
        return x.view(-1,1,300,5)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(300*5, 800), nn.ReLU(),
    nn.Linear(800, 400), nn.ReLU(),
    nn.Linear(400, 200), nn.ReLU(),
    nn.Linear(200, 100), nn.ReLU(),
    nn.Linear(100, 2))

def load_array(data_arrays,batch_size,is_train=True):
    dataset= data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

print("#### Now is loading data ####")
train_features=torch.load(train_features_path)
train_labels=torch.load(train_labels_path)
test_features=torch.load(test_features_path)
test_labels=torch.load(test_labels_path)
validation_features=torch.load(validation_features_path)
validaiton_labels=torch.load(validaiton_labels_path)
strict_features=torch.load(strict_features_path)
strict_labels=torch.load(strict_labels_path)

batch_size=65536
train_iter=load_array((train_features,train_labels),batch_size)
test_iter=load_array((test_features,test_labels),batch_size)
validation_iter=load_array((validation_features,validaiton_labels),batch_size)
strict_iter=load_array((strict_features,strict_labels),batch_size)
lr,num_epochs = [0.001,0.01,0.05,0.1],10000
print("#### Now is training ####")
for i in range(len(lr)):
    train(net, train_iter,test_iter,strict_iter,validation_iter,num_epochs, lr[i],file_name=lr[i])

