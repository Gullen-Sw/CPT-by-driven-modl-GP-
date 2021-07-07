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
q = 0
i = 0
u = 0
F = 0
w = 0
totaltime = 0
lis = []
pre_c=[]
interim_list=[]
timelist= []

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
study_time=CSVV_input.values[:, 0]

input_list = np.array([XX.ravel(), YY.ravel()]).T

#これでカーネルを定義します。odor plume用
kernel = sk_kern.RBF(length_scale=.5)

#alphaは発散しないように対角行列に加える値
m = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 80, normalize_y=True) 

start = time.time()
while study_time[i]!=None:

    output_list = CSVV_input.values[i, 1:46]
    output_list = output_list.reshape([-1,1])
    m.fit(input_list, output_list) #学習
    lis.append(study_time[i])
    output_list = output_list.ravel()
    
    for temp_y in range(0, 560, 10):
        for temp_x in range(-100, 110, 10):
            if temp_y in YY:
                if temp_x in XX: #座標点と一致するとき、そのまま入れる。

                    lis.append(output_list[u])
                    if u < 44:
                        u = u+1
                    else:
                        pass
                else:
                    pred = np.array([[temp_x],[temp_y]]).T
                    conc_mean, conc_std = m.predict(pred, return_std = True)
                    conc_pred = conc_mean
                    if conc_pred[0][0]<0:
                        conc_pred[0][0]=0
                    lis.append(conc_pred[0][0])
            else:
                pred = np.array([[temp_x], [temp_y]]).T
                conc_mean, conc_std = m.predict(pred, return_std = True)
                conc_pred = conc_mean
                if conc_pred[0][0]<0:
                    conc_pred[0][0]=0
                lis.append(conc_pred[0][0])
                               
    Result = np.array(lis)
    Result = Result.reshape(1,-1)
    if i==0:
        dataset = len(lis)
        sensor_matrix = np.empty((0, dataset), float)
    else:
        pass
    process_time = time.time() - start
    times = str(datetime.timedelta(seconds = process_time)).split(".")
    times = times[0]
    
    print('Dealing data : ', study_time[i], 'process time : ', times)
  
    sensor_matrix = np.append(sensor_matrix, Result, axis = 0) 

    i = i+1
    lis = []
    u = 0
    
    if study_time[i]==max(study_time):
        break
#finish

#extract to csv
df = pd.DataFrame(sensor_matrix)
df.to_csv('odor_GP_Result.csv', index = False, encoding = 'cp949') 
