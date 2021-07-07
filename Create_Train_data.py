# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:54:25 2021

@author: LEE
"""

import pandas as pd
import csv

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
#import GPy

from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor

Result = 'J-2_Done.csv'

#data sheet DeepLabCut
CSV = 'J-2_TrimDLC_resnet50_Obstacle_blackJun5shuffle1_1030000.csv'

#Odor Plume model(read odor concentration)

CSVV = 'J-2_conv.csv'

index = 0 #初めての時のみ1にして！

#check
print(CSV, "&&&", CSVV)

#pixel per mm
rate = 3.042

#init
init_x = 550
init_y = 180
csv_input = pd.read_csv(CSV,header=2)

#Test DataFrame
#csv_input.shape

i = 0
r = csv_input.values[:, 0]
list = []

with open(Result, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['temp_h_x_1', 'temp_h_x_2', 'h_study_delta_x', 'temp_h_y_1', 'temp_h_y_2', 'h_study_delta_y', 'temp_t_x_1', 'temp_t_x_2',  't_study_delta_x', 'temp_t_y_1', 'temp_t_y_2', ' t_study_delta_y'])	

            
while csv_input.values[i, 0]!=None:
#head x 2step & delta_x 
   temp_h_x_1 = (csv_input.values[i, 1] - init_x)/rate
   temp_h_x_2 = (csv_input.values[i+3, 1] - init_x)/rate
   h_study_delta_x=temp_h_x_2 - temp_h_x_1
 
#head y 2step & delta_y
   temp_h_y_1 = (csv_input.values[i, 2] - init_y)/rate
   temp_h_y_2 = (csv_input.values[i+3, 2] - init_y)/rate
   h_study_delta_y=temp_h_y_2 - temp_h_y_1

#tail x 2step & delta_x 
   temp_t_x_1 = (csv_input.values[i, 4] - init_x)/rate
   temp_t_x_2 = (csv_input.values[i+3, 4] - init_x)/rate
   t_study_delta_x=temp_t_x_2 - temp_t_x_1
  
#tail y 2step & delta_y
   temp_t_y_1 = (csv_input.values[i, 5] - init_y)/rate
   temp_t_y_2 = (csv_input.values[i+3, 5] - init_y)/rate
   t_study_delta_y=temp_t_y_2 - temp_t_y_1
   
#save File  
   list.extend([temp_h_x_1, temp_h_x_2, h_study_delta_x, temp_h_y_1, temp_h_y_2, h_study_delta_y, temp_t_x_1, temp_t_x_2, t_study_delta_x, temp_t_y_1, temp_t_y_2, t_study_delta_y])
   with open(Result, 'a', newline='') as f:
       writer = csv.writer(f)
       writer.writerow([list[0],list[1],list[2],list[3],list[4],list[5],list[6],list[7],list[8],list[9],list[10],list[11]])	
       
   i = i+1
   list = []
   if i + 2 == max(r):
       break

CSV = Result
csv_input = pd.read_csv(CSV,header=0)

#initialization
fig = plt.figure()
i = 0
u = 0


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

length = len(csv_input.values[:,0])

input_list = np.array([XX.ravel(), YY.ravel()]).T

#これでカーネルを定義します。odor plume用
kernel = sk_kern.RBF(length_scale=.5)

#alphaは発散しないように対角行列に加える値
m = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 80, normalize_y=True) 

start = time.time()
for i in range(length):
    u = round(3.3*i)
    output_list = CSVV_input.values[u, 1:46]
    output_list = output_list.reshape([-1,1])
    m.fit(input_list, output_list) #学習
    
    x_pos = csv_input.values[i, 0]
    y_pos = csv_input.values[i, 3]
    
    pred = np.array([[x_pos], [y_pos]]).T
    conc_mean, conc_std = m.predict(pred, return_std = True)
    conc_pred = conc_mean
    if conc_pred[0][0]<0:
        conc_pred[0][0]=0
        
  
    csv_input.loc[i,'Odor Value'] = conc_pred[0][0]

    if i > length-2 :pass
    else :
        csv_input.loc[i+1,'Odor Value+1'] = conc_pred[0][0]

    if i > length-3 : pass
    else : 
        csv_input.loc[i+2,'Odor Value+2'] = conc_pred[0][0]

    
    if i > length-4 :  pass
    else : 
        csv_input.loc[i+3,'Odor Value+3'] = conc_pred[0][0]

    
    process_time = time.time() - start
    times = str(datetime.timedelta(seconds = process_time)).split(".")
    times = times[0]
       
    print('Iteration : ', i+1,'/', length, 'process time : ', times)

    i = i+1
    u = 0
    
#finish

#extract to csv
csv_input = csv_input.fillna(0)
if index != 1:
    csv_input.to_csv('Train_data_set.csv', header = False, index = False, mode = 'a', encoding = 'cp949')
else:
    csv_input.to_csv('Train_data_set.csv', index = False, mode = 'w', encoding = 'cp949') 


