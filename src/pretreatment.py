import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
#!/usr/bin/python
def one_hot(sequence):
    sequence=sequence.upper()
    sequence=re.sub('[^ACGT]','Z',sequence)
    sequence_data_for_encode = np.array(list(sequence))
    sequence_data_for_encode=sequence_data_for_encode.reshape(-1,1)
    model = [['A'],['G'],['C'],['T'],['Z']]
    enc = OneHotEncoder()
    enc.fit(model)
    enc.transform(model).toarray()
    max_len = 300
    sequence_matrix = np.zeros((1,max_len*5))
    for i in range(len(sequence)):
        tempdata = enc.transform(sequence_data_for_encode).toarray()
        additional = np.zeros((1,max_len*5 - np.size(tempdata)))
        sequence_matrix = np.hstack((tempdata.flatten(),additional.flatten()))
    return sequence_matrix.reshape(300,5)

def file_to_one_hot(file_path,positive=True):
    with open (file_path) as file:
        data=file.readlines()
    features=[]
    labels=[]
    if positive==True:
        b=1
    else:
        b=0
    for i in range(len(data)):
        if len(data[i].strip())<=300:
            a=one_hot(data[i].strip())
            features.append(a)
            labels.append(b)
    features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.25,random_state=33)
    return torch.from_numpy(np.array(features_train)),torch.from_numpy(np.array(labels_train)),torch.from_numpy(np.array(features_test)),torch.from_numpy(np.array(labels_test))


def vfile_to_one_hot(file_path,positive=True):
    with open (file_path) as file:
        data=file.readlines()
    features=[]
    labels=[]
    if positive==True:
        b=1
    else:
        b=0
    for i in range(len(data)):
        if len(data[i].strip())<=300:
            a=one_hot(data[i].strip())
            features.append(a)
            labels.append(b)
    return torch.from_numpy(np.array(features)),torch.from_numpy(np.array(labels))


#uninverton_dataset
file_path="../data/inverton_sequence.txt"
features_train,labels_train=vfile_to_one_hot(file_path,positive=True)
#inverton_dataset
# positive_file_path='/data4/wenjiejie/Graduation_Project/add_data/model/data/extend/remove_extend2/extend_inverton.txt'
# features2_train,labels2_train=vfile_to_one_hot(positive_file_path,positive=True)

# train_features=torch.cat((features_train,features2_train),dim=0).to(torch.float32)
# train_labels=torch.cat((labels_train,labels2_train),dim=0).long()
end=time.time()
print(end-start)
torch.save(features_train,r'/data4/wenjiejie/Graduation_Project/add_data/analysis/extend_data/extend_inverton/true_train_features.pt')
torch.save(labels_train,r'/data4/wenjiejie/Graduation_Project/add_data/analysis/extend_data/extend_inverton/true_train_labels.pt')

# torch.save(features2_train,r'/data4/wenjiejie/Graduation_Project/add_data/model/data/extend/remove_extend2/true_train_features.pt')
# torch.save(labels2_train,r'/data4/wenjiejie/Graduation_Project/add_data/model/data/extend/remove_extend2/true_train_labels.pt')

