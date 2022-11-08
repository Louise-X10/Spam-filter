#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:01:32 2022

@author: liuyilouise.xu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split

data  = pd.read_csv("SpamInstances.txt", header=0, delimiter=' ', names=['index','label','data'])

#trial_data  = pd.read_csv("trial.txt", header=0, delimiter=' ', names=['index','label','x','y','z'])
# X1: warm = Y, cold = N
# X2: yes = Y, no = N
# X3: Wet = Y, Dry = N
# Y: fast = 1, slow = -1

# nbc = matrix of conditional probabilities
# # 1st col = P(Y | label = 1)
# # 2nd col = P(N | label = 1)
# # 3rd col = P(Y | label = -1)
# # 4th col = P(N | label = -1)
def build_naive_bayes_classifier(data):
    data_len = len([*data['data'].iloc[0]]) #decompose features into list
    nbc = np.zeros(shape=(data_len, 4))
    for i in range(data.shape[0]):
        #cnts = [data.iloc[i]['x'],data.iloc[i]['y'],data.iloc[i]['z']]
        #inv_cnts = [not cnt for cnt in cnts]
        
        cnts = [*data['data'].iloc[i]]
        cnts = list(map(int, cnts))
        inv_cnts = [not cnt for cnt in cnts]
        inv_cnts = list(map(int, inv_cnts))
        
        
        if data['label'].iloc[i] == 1:
            nbc[:,0] += cnts
            nbc[:,1] += inv_cnts
        else:
            nbc[:,2] += cnts
            nbc[:,3] += inv_cnts
    
    pos_cnt = np.sum(data['label'] == 1)
    neg_cnt = data.shape[0]-pos_cnt
    
    #nbc_old = nbc
    # laplace smoothing by 1: (nbc_cnt + 1) / (total_cnt + 1*feature_num)
    alpha = 1
    nbc = np.add(nbc, alpha)
    nbc[:,0] = np.true_divide(nbc[:,0], pos_cnt+data_len*alpha)
    nbc[:,1] = np.true_divide(nbc[:,1], pos_cnt+data_len*alpha)
    nbc[:,2] = np.true_divide(nbc[:,2], neg_cnt+data_len*alpha)
    nbc[:,3] = np.true_divide(nbc[:,3], neg_cnt+data_len*alpha)
    
    pos_prob = pos_cnt/data.shape[0]

    return nbc, pos_prob
        
def classify(data, nbc, pos_prob):
    preds = np.zeros(data.shape[0])
    #underflow = 0
    for i in range(data.shape[0]):
        cnts = [*data['data'].iloc[i]]
        cnts = list(map(int, cnts))
        cnts = list(map(bool, cnts))
        inv_cnts = [not cnt for cnt in cnts]
        probs = nbc[cnts,:][:, [0,2]] # P(Y|1) and P(Y|-1)
        inv_probs = nbc[inv_cnts,:][:, [1,3]] # P(N|1) and P(N|-1)
        
        # product of probabilities
        #pos = (np.prod(probs[:, 0])*np.prod(inv_probs[:, 0]))*(pos_prob)
        #neg = (np.prod(probs[:, 1])*np.prod(inv_probs[:, 1]))*(1-pos_prob)
        
        # use sum of log probabilities to prevent numeric underflow
        pos = (np.sum(np.log(probs[:, 0]))+ np.sum(np.log(inv_probs[:, 0])))+np.log(pos_prob)
        neg = (np.sum(np.log(probs[:, 1])) + np.sum(np.log(inv_probs[:, 1])))+np.log(1-pos_prob)
        #print(pos, neg)
        

        pred = 1 if pos >= neg else -1
        preds[i] = pred
        '''
        if (pos == 0 and neg == 0):
            underflow += 1
            print(f"Both underflow! Predict positive {pred}")
        
        '''
    tp = sum(np.logical_and(preds == 1, data['label'] == 1))
    fp = sum(np.logical_and(preds == 1, data['label'] == -1))
    
    # if no positive predictions, precision of predicting pos correctly is 1
    precision = tp / sum(preds == 1)# tp / (tp + fp) = tp/predict p
    #print(f'tp = {tp}, predict p = {sum(preds==1)}, precision = {precision}')
    #print(f' predict p for size {data.shape[0]} = {sum(preds==1)}')

    recall = tp / sum(data['label'] == 1)  # tp / (tp + fn) = tp / actual p, true positive rate
    #print(f"tp = {tp}, p = {sum(data['label'] == 1)}, recall = {recall}")

    fpr = fp / sum(data['label'] == -1)# fp / (fp+tn) = fp / n = false positive rate
    #print(f"fp = {fp}, n = {sum(data['label'] == -1)}, fpr = {fpr}")

    return precision, recall, fpr


pos_data = data[data['label'] == 1]
neg_data = data[data['label'] == -1]

precisions = []
recalls = []

partition_sizes = []
for i in range(1, 20):
    partition_sizes.append(i*100)
partition_sizes.append(data.shape[0])

for partition_size in partition_sizes:    
    pos_idx = np.random.randint(pos_data.shape[0], size = partition_size//2)
    neg_idx = np.random.randint(neg_data.shape[0], size = partition_size//2)
    # equally represented data
    partition_data = pd.concat([pos_data.iloc[pos_idx, :], neg_data.iloc[neg_idx, :]])
    
    train_data, test_data  = train_test_split(partition_data, test_size = 0.2)
    
    nbc, pos_prob = build_naive_bayes_classifier(train_data)
    precision, recall, fpr = classify(test_data, nbc, pos_prob)
    
    precisions.append(precision)
    recalls.append(recall)
   # print(f"Underflow for size {partition_size*0.2} is ", underflow)

# rpc curve = tpr or recall vs fpr

fig = plt.figure()
plt.xlim(1.1, -0.1)
plt.ylim(-0.1, 1.1)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('ROC curve')
plt.plot([1.1, -0.1], [-0.1, 1.1],'b--')

# create static plot
plt.scatter(precisions, recalls)
'''
# create animated plot
graph, = plt.plot([], [], 'o')

def animate(i):
    graph.set_data(precisions[:i+1], recalls[:i+1])
    return graph

ani = FuncAnimation(fig, animate, frames=20, interval=500)
'''
plt.show()

