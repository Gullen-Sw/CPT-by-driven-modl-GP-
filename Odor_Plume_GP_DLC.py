# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:22:16 2021

@author: LEE
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
#import GPy

from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor

#DeepLabCut　―　キャリブレーション済ファイル読み込み
CSV = 'A-1_Done.csv'
csv_input = pd.read_csv(CSV,header=0)
length=len(csv_input.values[:, 0])

#障害物あるかないか↓ここで設定
#if obstract exists, "obst"=1, otherwise "obst" =0 
obst = 1


#Odor Plume model(read odor concentration)
if obst == 0:
	CSVV = 'normal1_suko.csv'
	print("障害物なし")
elif obst == 1:
	CSVV = 'A-1_conv.csv'
	print("障害物あり")

#initialization
fig = plt.figure()
i = 0
lis = []

plt.clf()

#全体座標作成
x=[-100.0,-50.0,0.0,50.0,100.0]
y=[0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0]
xx,yy = np.meshgrid(x,y)
#センサ格子作成
X=[-100.0,-50.0,0.0,50.0,100.0]
Y=[50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0]
XX,YY = np.meshgrid(X,Y)

CSVV_input = pd.read_csv(CSVV,header=None)

input_list = np.array([XX.ravel(), YY.ravel()]).T

#これでカーネルを定義します。odor plume用
kernel = sk_kern.RBF(length_scale=.5)

#alphaは発散しないように対角行列に加える値
m = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 80, normalize_y=True) 

start = time.time()
for i in range(length):

    output_list = CSVV_input.values[i, 1:46]
    output_list = output_list.reshape([-1,1])
    m.fit(input_list, output_list) #学習
    lis.append(i)
    
    x_pos = csv_input.values[i, 0]
    y_pos = csv_input.values[i, 3]
    
    pred = np.array([[x_pos], [y_pos]]).T
    conc_mean, conc_std = m.predict(pred, return_std = True)
    conc_pred = conc_mean
    if conc_pred[0][0]<0:
        conc_pred[0][0]=0
    lis.append(conc_pred[0][0])
                               
    Result = np.array(lis) #list と float型が混ざっててエラー
    Result = Result.reshape(1,-1)
    if i==0:
        dataset = len(lis)
        sensor_matrix = np.empty((0, dataset), float)
    else:
        pass
    process_time = time.time() - start
    times = str(datetime.timedelta(seconds = process_time)).split(".")
    times = times[0]
    
    print('Iteration : ', i+1, '/', length, 'process time : ', times)
  
    sensor_matrix = np.append(sensor_matrix, Result, axis = 0) 

    i = i+1
    lis = []

#finish

#extract to csv
df = pd.DataFrame(sensor_matrix)
df.to_csv('odor_GP_Result_DLC.csv', index = False, encoding = 'cp949') 
