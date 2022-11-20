# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 02:25:59 2022

@author: Shajib Ghosh
"""
"""Importing the dependencies"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

"""Constant Declaration"""
k = 25 
ts = float(input("Enter test data size (e.g., 0.2 for 20% test data): "))    
seed = 2022
n_est = 10000

"""Main Calculation"""
img_id = []
accuracy = []
precision = []
recall = []
fScore = []
for id in tqdm(range(49,54)):  
    id = str(id)
    df = pd.read_csv(r'/home/UFAD/shajib.ghosh/pcb-comp-feature-selection/extracted_features/K' + str(k) + '_' + id + '.csv', index_col=0)
    df = df.fillna(0) #'nan'--> 0
    data  = np.load(r'/home/UFAD/shajib.ghosh/pcb-comp-feature-selection/bbox_gt/' + id + '_bboxGT.npy')[:,1]
    df['GT'] = data 

    feat_labels = df.columns.tolist()
    feat_labels = feat_labels[:-1]
    feat_labels = feat_labels[1:]

    X = df[feat_labels].values
    y = df['GT'].values
    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=ts,random_state=seed)

    # Instantiate model 
    rf=RandomForestClassifier(n_estimators=n_est,n_jobs=-1,random_state=seed)
     
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    fscore = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nGenerating results for image id : {id}.png")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1-Score: {fscore}")

    img_id.append(id)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    fScore.append(fscore)

    importances=rf.feature_importances_
    indices=np.argsort(importances)[::-1] 
    
    list_feat = []
    list_imp_score = []
    
    for f in range(X_train.shape[1]):
        list_feat.append(feat_labels[indices[f]])
        list_imp_score.append(importances[indices[f]])
        
    c={"feature" : list_feat, "importance" : list_imp_score}
    feat_sel_data = pd.DataFrame(c)
    feat_sel_data.to_csv(r'/home/UFAD/shajib.ghosh/pcb-comp-feature-selection/Supervised/Embedded/Results/RF/'+'RF_C_K'+str(k)+'_'+id+'.csv', index = None)
    print(f"Calculation completed for image id: {id}.png")

dictionary = {'image_id': img_id, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1-score': fScore}  
df_results = pd.DataFrame(dictionary) 
df_results.to_csv(r'/home/UFAD/shajib.ghosh/pcb-comp-feature-selection/Supervised/Embedded/Results/'+ 'RF_C_summary.csv')
print("Process completed for Feature Selection using Random Forest Classifier.")