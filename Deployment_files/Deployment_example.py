#!/usr/bin/env python
# coding: utf-8

# imports of libraries


import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


# Read the data

train_data = pd.read_csv(r'C:\Users\hp 15\Downloads\train.csv (1)\train.csv', index_col = ['Entry_id'])
test_data = pd.read_csv(r'C:\Users\hp 15\Downloads\test (1).csv', index_col = ['Entry_id'])

#data preprocessing

y = train_data['e_signed']
train_data.drop('e_signed', axis = 1, inplace = True)
train_x, valid_x, train_y, valid_y = train_test_split(train_data,y,test_size = 0.2, random_state = 1)

def train(train_data, y):
    #dicts = train_data[columns].to_dict(orient='records')
    #dv = DictVectorizer(sparse=False)
    #X = dv.fit_transform(dicts)
    train_data['full_months_employed'] = 12*train_data['years_employed']+train_data['months_employed']
    encoder = LabelEncoder()
    for col in train_data.columns:
        if train_data[col].dtype == 'object':
            train_data[col] = encoder.fit_transform(train_data[col])
    model = CatBoostClassifier(iterations = 1500, learning_rate = 0.02, depth = 4)
    model.fit(train_data, y)
    return model

def predict(test_data, model):
    #dicts = test_data[columns].to_dict(orient='records')
    #X = dv.transform(dicts)
    encoder = LabelEncoder()
    test_data['full_months_employed'] = 12*test_data['years_employed']+test_data['months_employed']
    for col in test_data.columns:
        if test_data[col].dtype == 'object':
            test_data[col] = encoder.fit_transform(test_data[col])
    y_pred = model.predict_proba(test_data)[:,1]
    return y_pred

model = train(train_x, train_y)
y_pred = predict(valid_x, model)
auc = roc_auc_score(valid_y, y_pred)
auc
print('The auc score obtained is:{}'.format(auc))

model = train(train_data, y)
y_pred = predict(test_data, model)

#saving the final model

output_file = 'final_model.bin'


with open(output_file, 'wb') as f_out: 
    pickle.dump(model, f_out)

print('This model has been successfully saved to: {}'.format(output_file))






