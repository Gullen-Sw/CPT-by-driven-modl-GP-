#参考HP
#https://nykergoto.hatenablog.jp/entry/2017/05/29/python%E3%81%A7%E3%82%AC%E3%82%A6%E3%82%B9%E9%81%8E%E7%A8%8B%E5%9B%9E%E5%B8%B0_~_%E3%83%A2%E3%82%B8%E3%83%A5%E3%83%BC%E3%83%AB%E3%81%AE%E6%AF%94%E8%BC%83_~
#https://funatsu-lab.github.io/open-course-ware/machine-learning/gaussian-process/
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import GPy
import math
import os
import cv2
import matplotlib.patches as pat
import csv
import time
from time import sleep
import random
from statistics import mean
from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor


#推定したカイコガ位置での推定濃度値とカイコガの進行ΔxΔyを学習データにして行動シミュレーション---------------------------------------

#study_model
#-------------------------------------------------------------------
CSV = 'Test.csv'
#-------------------------------------------------------------------

csv_input = pd.read_csv(CSV,header=0)

#study_output:displacement of position (4step - 1step)
study_delta_head_x=csv_input.values[:-1, 9]
study_delta_head_y=csv_input.values[:-1, 12]
study_delta_tail_x=csv_input.values[:-1, 15]
study_delta_tail_y=csv_input.values[:-1, 18]

#study_input:current voltage
study_conc1=csv_input.values[:-1,20]
#study_input:voltage after 1 step
study_conc2=csv_input.values[:-1,21]
#study_input:voltage after 2 step
study_conc3=csv_input.values[:-1,22]
#study_input:voltage after 3 step
study_conc4=csv_input.values[:-1,23]


#四次元入力、一次元出力_study用
study_input_list = np.array([study_conc1.ravel(), study_conc2.ravel(), study_conc3.ravel(), study_conc4.ravel()]).T

study_output_list_head_x = study_delta_head_x.reshape(len(csv_input)-1,1)
study_output_list_head_y = study_delta_head_y.reshape(len(csv_input)-1,1)

study_output_list_tail_x = study_delta_tail_x.reshape(len(csv_input)-1,1)
study_output_list_tail_y = study_delta_tail_y.reshape(len(csv_input)-1,1)


print('optimization')
#gaussianのモデル作成
#-------------------------------------------------------------------
#Kernel function
kernel = sk_kern.RBF(length_scale=.5)+sk_kern.WhiteKernel()

#optimization (alphaは発散しないように対角行列に加える値,) 
Mx = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 80, normalize_y=True)
My = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 80, normalize_y=True)
Mx_tail = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 80, normalize_y=True)
My_tail = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 80, normalize_y=True)

Mx.fit(study_input_list, study_output_list_head_x)
My.fit(study_input_list, study_output_list_head_y)
Mx_tail.fit(study_input_list, study_output_list_tail_x)
My_tail.fit(study_input_list, study_output_list_tail_y)


#--------------------------------------------------------------------
#record the parameter
kkk = Mx.log_marginal_likelihood()
kkkk = My.log_marginal_likelihood()

sss = Mx_tail.log_marginal_likelihood()
ssss = My_tail.log_marginal_likelihood()

params_x_head = Mx.kernel_.get_params()
params_y_head = My.kernel_.get_params()

params_x_tail = Mx_tail.kernel_.get_params()
params_y_tail = My_tail.kernel_.get_params()

print(kkk)
print(kkkk)
print(sss)
print(ssss)

print(params_x_head)
print(params_y_head)
print(params_x_tail)
print(params_y_tail)

with open('param.txt', 'w') as f:
	print('Mx_head : log_marginal_likelihood', file=f)
	print(kkk,params_x_head, file=f)
	print('My_head : log_marginal_likelihood', file=f)
	print(kkkk,params_y_head, file=f)
	print('Mx_tail : log_marginal_likelihood', file=f)
	print(sss,params_x_tail, file=f)
	print('My : log_marginal_likelihood', file=f)
	print(ssss,params_y_tail, file=f)
#-------------------------------------------------------------------

#finish sign
print('bujide')



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

CSVV_input = pd.read_csv(CSVV,header=None)
conc = CSVV_input.values[:, 1:46]
study_time=CSVV_input.values[:, 0]

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



#inipos=探索開始の初期位置
for inipos in range (-100,150,50):
	SuccessRate=0
	success=0
	totaltime = 0
	avetime = 0
	for ite in range(100): #iteration number
		#initial_Pos--------------------------------------------
		init_x = np.array([inipos])
		init_y = np.array([500])
		#-------------------------------------------------------
		name1 = '(' + str(init_x[0]) + ',' + str(init_y[0]) + ')' + '_simulation_success.csv'
		name2 = '(' + str(init_x[0]) + ',' + str(init_y[0]) + ')_' + str(ite) + '.csv'
		F=0
		print('---------------------------')
		print('simulation_number=' + str(ite))

		for sim in range(len(study_time)): #time limit
			if F == 0:
				plt.clf()
				#二次元入力、一次元出力_現場位置の濃度用
				input_list = np.array([XX.ravel(), YY.ravel()]).T
				output_list = conc[q,:]
				output_list = output_list.reshape([-1,1])
				
				#これでカーネルを定義します。odor plume用
				kernel = sk_kern.RBF(length_scale=.5)
				#alphaは発散しないように対角行列に加える値
				m = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 80, normalize_y=True)
				m.fit(input_list, output_list)
				pred = np.array([init_x.ravel(), init_y.ravel()]).T
				conc_mean, conc_std = m.predict(pred, return_std=True)
				print('conc_mean = ' + str(conc_mean) + ',    conc_std = ' + str(conc_std))			
				conc_pred = conc_mean + (conc_std * random.uniform(-1,1))

				print('カイコガ位置の推定濃度値fromガウス過程 = ' + str(conc_pred[0]))
				pre_c.append(conc_pred[0])
				
     	
				if sim > 2:
					pred_x = np.array([pre_c[q], pre_c[q-1], pre_c[q-2], pre_c[q-3]])
					pred_y = np.array([pre_c[q], pre_c[q-1], pre_c[q-2], pre_c[q-3]])
				elif sim == 0:
					pred_x = np.array([pre_c[q], [0], [0], [0]])
					pred_y = np.array([pre_c[q], [0], [0], [0]])
				elif sim == 1:
					pred_x = np.array([pre_c[q], pre_c[q-1], [0], [0]])
					pred_y = np.array([pre_c[q], pre_c[q-1], [0], [0]])
				elif sim == 2:
					pred_x = np.array([pre_c[q], pre_c[q-1], pre_c[q-2], [0]])
					pred_y = np.array([pre_c[q], pre_c[q-1], pre_c[q-2], [0]])
				print(pred_x)
				print(pred_y)


				x_mean, x_std = Mx.predict(pred_x.reshape(1, -1), return_std=True)
				y_mean, y_std = My.predict(pred_y.reshape(1, -1), return_std=True)
				print('x_mean = ' + str(x_mean) + ',    x_std = ' + str(x_std))			
				print('y_mean = ' + str(y_mean) + ',    y_std = ' + str(y_std))
				out_x = x_mean + (x_std * random.uniform(-1,1))
				out_y = y_mean + (y_std * random.uniform(-1,1))

		
				print('Δx = ' + str(out_x))
				print('Δy = ' + str(out_y))


				if (init_x + out_x[0][0]) > 100:
					init_x = np.array([99.9])
					w=1
				if (init_x + out_x[0][0]) < -100:
					init_x = np.array([-99.9])
					w=1
				if (init_y + out_y[0][0]) > 550:
					init_y = np.array([549.9])
					w=1
				if (init_y + out_y[0][0]) < 0:
					init_y = np.array([0.1])
					w=1
#--------------------------------------------------------------
				if obst == 1:  #障害物がある場合のみ
					if (((init_y + out_y[0][0])-250)*((init_y + out_y[0][0])-250) + (init_x + out_x[0][0])*(init_x + out_x[0][0])) < 400:
						w=1
#--------------------------------------------------------------
				if w == 0:
					init_x = init_x + out_x[0][0]
					init_y = init_y + out_y[0][0]	
	
			# print(q)
				print(w)
				print('カイコガPos_x = ' + str(init_x))
				print('カイコガPos_y = ' + str(init_y))
				#CSV吐き出し
				lis.extend([study_time[q],init_x[0],init_y[0],conc_pred[0][0]])
				#CSV吐き出し:time,x,y,conc
				with open(name2, 'a', newline='') as f:
					writer = csv.writer(f)	
					writer.writerow([lis[0],lis[1],lis[2],lis[3]])		

				print(q)
				q=q+1
				w=0
				conc_pred=0
				out_x=0
				out_y=0
				lis = []
				# for making time to write data into csv 
				sleep(0.1)
				
				if sim == (len(study_time)-1):
					q=0
					lis = []
					conc_pred=0
					pre_c=[]
					out_x=0
					out_y=0                
					break
				
				#終了条件
				if init_x*init_x + init_y*init_y < 2500:	
					timelist = CSVV_input.values[q-1, 0]
					totaltime = totaltime + timelist
					q=0
					success = success + 1
					lis = []
					pre_c=[]
					conc_pred=0
					out_x=0
					out_y=0                
					break

	#CSV吐き出し:SuccessRate	
	SuccessRate=float(success)/(float(ite)+1.0)
	avetime = totaltime/(float(ite)+1.0)
	print('average time='+str(avetime))
	print('success_rate='+str(SuccessRate))
	with open(name1, 'a', newline='') as ff:
		writer = csv.writer(ff)
		writer.writerow([SuccessRate,avetime])
#----------------------------------------------------------------------------------------------------------------------------



#movie
#--------------------------------------------------------	
		# plt.xlabel('y [mm]')
		# plt.ylabel('x [mm]')
		# # plt.xlimでグラフの幅指定
		# plt.xlim([-100,100])
		# plt.ylim([0,550])
		# # カイコガ位置を出現。sがプロットの大きさ指定。alphaが透過性。
		# plt.scatter(init_x, init_y, s=25, edgecolors="black",facecolor='None',marker='o',alpha=0.5)
		# #センサ格子の位置を出現。sがプロットの大きさ指定。alphaが透過性。
		# plt.scatter(XX, YY, s=50, edgecolors="black",facecolor='None',marker='s',alpha=0.5)
		# # 時間出現(x,y,表示文字)
		# time = study_time[q]
		# plt.text(-100,580,"T = " + str(time) + " s")	
		
		# # goal半径出現
		# ww = np.arange(-math.pi , math.pi, 0.01)
		# for ww in ww:
			# plt.scatter(50*math.cos(ww) , 50*math.sin(ww) , c='black', s = 0.1)	
		
		# # startPos出現
		# plt.scatter(0 , 500 , c='black', s = 10)
	
		# # アスペクト比設定
		# plt.axes().set_aspect('equal')
#--------------------------------------------------------