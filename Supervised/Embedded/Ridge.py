# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 20:32:51 2022

@author: Shajib Ghosh
"""
"""Importing the dependencies"""
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

"""Constant Declaration"""
k = 25 
ts = float(input("Enter test data size (e.g., 0.2 for 20% test data): "))    
seed = 2022
n_est = 1000

"""Main Calculation"""
img_id = []
mse = []
mae = []
R2_Score = []
for id in tqdm(range(49,64)):  
    id = str(id)
    df = pd.read_csv(r'/home/UFAD/shajib.ghosh/CompStat_Feature_Selection/extracted_features/K' + str(k) + '_' + id + '.csv', index_col=0)
    df = df.fillna(0) #'nan'--> 0
    data  = np.load(r'/home/UFAD/shajib.ghosh/CompStat_Feature_Selection/bbox_gt/' + id + '_bboxGT.npy')[:,1]
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
    ridge = RidgeCV(alphas=np.arange(0.00001, 1, 0.05), cv=10)
    ridge.fit(X, y)
    print("\nBest alpha value: ", ridge.alpha_)
    RidgeReg = Ridge(alpha=ridge.alpha_, max_iter=n_est)
    RidgeReg.fit(X_train,y_train) 
    y_pred = RidgeReg.predict(X_test)
    mean_sqr_err = mean_squared_error(y_test, y_pred)
    mean_abs_err = mean_absolute_error(y_test, y_pred)
    r2_Score = r2_score(y_test, y_pred)

    print(f"\nGenerating results for image id : {id}.png")
    print(f"Mean squared error (MSE): {mean_sqr_err}")
    print(f"Mean absolute error (MAE): {mean_abs_err}")
    print(f"R2 Score: {r2_Score}")

    img_id.append(id)
    mse.append(mean_sqr_err)
    mae.append(mean_abs_err)
    R2_Score.append(r2_Score)

    importances=np.abs(RidgeReg.coef_)
    indices=np.argsort(importances)[::-1] 
    
    list_feat = []
    list_imp_score = []
    
    for f in range(X_train.shape[1]):
        list_feat.append(feat_labels[indices[f]])
        list_imp_score.append(importances[indices[f]])
        
    c={"feature" : list_feat, "importance" : list_imp_score}
    feat_sel_data = pd.DataFrame(c)
    feat_sel_data.to_csv(r'/home/UFAD/shajib.ghosh/CompStat_Feature_Selection/Supervised/Embedded/Results/Ridge/'+'Ridge_K'+str(k)+'_'+id+'.csv', index = None)
    print(f"Calculation completed for image id: {id}.png")

dictionary = {'image_id': img_id, 'mean_squared_error': mse, 'mean_absolute_error': mae, 'r2_score': R2_Score}  
df_results = pd.DataFrame(dictionary) 
df_results.to_csv(r'/home/UFAD/shajib.ghosh/CompStat_Feature_Selection/Supervised/Embedded/Results/'+ 'Ridge_Regression_summary.csv')
print("Process completed for Feature Selection using Ridge Regression.")