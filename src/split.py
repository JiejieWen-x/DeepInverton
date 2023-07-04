import os
from sklearn.model_selection import KFold
import torch

#split dataset
def kfold(n_splits=4,features=[],labels=[],label='true',path=r''):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    train_fold_index=[]
    test_fold_index=[]
    for train_index , test_index in kf.split(range(len(features))):
        train_fold_index.append(train_index)
        test_fold_index.append(test_index)
    outpath=os.path.join(path,'kfold')
    if os.path.exists(outpath) is False:
        os.mkdir(outpath)
    for i in range(len(test_fold_index)):
        #train features
        exec ("%s_train_fold_%s=train_fold_index[i]"%(label,i))
        exec("%s_train_features_fold_%s=features[[train_fold_index[i]]]"%(label,i))
        exec("path_%s=os.path.join(outpath,'%s_train_features_fold_'+str(i))+'.pt'"%(i,label))
        exec("torch.save(%s_train_features_fold_%d,path_%s)"%(label,i,i))
        #labels features
        exec("%s_train_labels_fold_%s=labels[[train_fold_index[i]]]"%(label,i))
        exec("path_%s=os.path.join(outpath,label+'_train_labels_fold_'+str(i))+'.pt'"%i)
        exec("torch.save(%s_train_labels_fold_%d,path_%s)"%(label,i,i))
        #test feature
        exec ("%s_test_fold_%s=test_fold_index[i]"%(label,i))
        exec("%s_test_features_fold_%s=features[[test_fold_index[i]]]"%(label,i))
        exec("path_%s=os.path.join(outpath,label+'_test_features_fold_'+str(i))+'.pt'"%i)
        exec("torch.save(%s_test_features_fold_%d,path_%s)"%(label,i,i))
        #test label
        exec("%s_test_labels_fold_%s=labels[[test_fold_index[i]]]"%(label,i))
        exec("path_%s=os.path.join(outpath,label+'_test_labels_fold_'+str(i))+'.pt'"%i)
        exec("torch.save(%s_test_labels_fold_%d,path_%s)"%(label,i,i))

def true_false(true_features=[],true_labels=[],false_features=[],false_label=[]):
    features=torch.cat((true_features,false_features),dim=0).to(torch.float32)
    labels=torch.cat((true_labels,false_label),dim=0).long()
    return features,labels

def train_test(path=r'',n_split=5):
    outpath=os.path.join(path,'train_test')
    if os.path.exists(outpath) is False:
        os.mkdir(outpath)
    for i in range(0,n_split):
        print(i)
        exec("true_train_features_path_%s=os.path.join(path,'true_train_features_fold_'+str(i)+'.pt')"%(i))
        #print(true_train_features_path_1)
        exec("true_train_features%s=torch.load(true_train_features_path_%s)"%(i,i))
        exec("true_train_labels_path_%s=os.path.join(path,'true_train_labels_fold_'+str(i)+'.pt')"%(i))
        exec("true_train_labels%s=torch.load(true_train_labels_path_%s)"%(i,i))

        exec("true_test_features_path_%s=os.path.join(path,'true_test_features_fold_'+str(i)+'.pt')"%(i))
        exec("true_test_features%s=torch.load(true_test_features_path_%s)"%(i,i))
        exec("true_test_labels_path_%s=os.path.join(path,'true_test_labels_fold_'+str(i)+'.pt')"%(i))
        exec("true_test_labels%s=torch.load(true_test_labels_path_%s)"%(i,i))

        exec("false_train_features_path_%s=os.path.join(path,'false_train_features_fold_'+str(i)+'.pt')"%(i))
        exec("false_train_features%s=torch.load(false_train_features_path_%s)"%(i,i))
        exec("false_train_labels_path_%s=os.path.join(path,'false_train_labels_fold_'+str(i)+'.pt')"%(i))
        exec("false_train_labels%s=torch.load(false_train_labels_path_%s)"%(i,i))

        exec("false_test_features_path_%s=os.path.join(path,'false_test_features_fold_'+str(i)+'.pt')"%(i))
        exec("false_test_features%s=torch.load(false_test_features_path_%s)"%(i,i))
        exec("false_test_labels_path_%s=os.path.join(path,'false_test_labels_fold_'+str(i)+'.pt')"%(i))
        exec("false_test_labels%s=torch.load(false_test_labels_path_%s)"%(i,i))

        exec("train_features_fold_%s,train_labels_fold_%s=true_false(true_train_features%s,true_train_labels%s,false_train_features%s,false_train_labels%s)"%(i,i,i,i,i,i))
        exec("path_%s=os.path.join(outpath,'train_features_fold_'+str(i))+'.pt'"%(i))
        exec("torch.save(train_features_fold_%d,path_%s)"%(i,i))
        exec("path_%s=os.path.join(outpath,'train_labels_fold_'+str(i))+'.pt'"%(i))
        exec("torch.save(train_labels_fold_%d,path_%s)"%(i,i))

        exec("test_features_fold_%s,test_labels_fold_%s=true_false(true_test_features%s,true_test_labels%s,false_test_features%s,false_test_labels%s)"%(i,i,i,i,i,i))
        exec("path_%s=os.path.join(outpath,'test_features_fold_'+str(i))+'.pt'"%(i))
        exec("torch.save(test_features_fold_%d,path_%s)"%(i,i))
        exec("path_%s=os.path.join(outpath,'test_labels_fold_'+str(i))+'.pt'"%(i))
        exec("torch.save(test_labels_fold_%d,path_%s)"%(i,i))
    

#split train and test dataset
positive_features=torch.load(r'../data/positive_train_features.pt')
positive_labels=torch.load(r'../data/positive_train_labels.pt')
negative_features=torch.load(r'../data/negative_train_features.pt')
negative_labels=torch.load(r'../data/negative_train_labels.pt')
positive_dir_path=r'../data/'
negative_dir_path=r'../data/'
kfold(n_splits=4,features=positive_features,labels=positive_labels,label='true',path=positive_dir_path)
kfold(n_splits=4,features=negative_features,labels=negative_labels,label='true',path=negative_dir_path)
train_test(path=r'../data/kfold/',n_split=5)

