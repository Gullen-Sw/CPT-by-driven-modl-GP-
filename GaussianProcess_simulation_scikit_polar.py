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
# import cv2
import matplotlib.patches as pat
import csv
import time
from time import sleep
import random
from statistics import mean
from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor
##########################
# SAVE-LOAD using joblib #
##########################
import joblib


#推定したカイコガ位置での推定濃度値とカイコガの進行ΔxΔyを学習データにして行動シミュレーション--------------------------------------------------
# # #Setting-----------------------------------------------------
# #移動倍率
# alpha = 1
# #移動平均ステップ
# step = 30
# #入力の次元（input_dim）setting
# input_dim = 2
# input_dim_plume = 2
# #-----------------------------------------------------

#Load model? 0:Not read, 1:Read
readM = 0
#study_model
#-------------------------------------------------------------------
CSV = 'Study_gaussian_new_new_polar.csv'
#-------------------------------------------------------------------

csv_input = pd.read_csv(CSV,header=None)
#study_output:displacement of position (4step - 1step)
study_delta_x=csv_input.values[:, 5]
study_delta_y=csv_input.values[:, 6]
#study_input:current voltage
study_conc1=csv_input.values[:,7]
#study_input:voltage after 1 step
study_conc2=csv_input.values[:,8]
#study_input:voltage after 2 step
study_conc3=csv_input.values[:,9]
#study_input:voltage after 3 step
study_conc4=csv_input.values[:,10]


#四次元入力、一次元出力_study用
study_input_list = np.array([study_conc1.ravel(), study_conc2.ravel(), study_conc3.ravel(), study_conc4.ravel()]).T

study_output_list_x = study_delta_x.reshape(len(csv_input),1)
study_output_list_y = study_delta_y.reshape(len(csv_input),1)

print('bujidesu')

if readM == 0:
	#gaussianのモデル作成
	#-------------------------------------------------------------------
	kernel = sk_kern.RBF(length_scale=.5)+sk_kern.WhiteKernel()
	#alphaは発散しないように対角行列に加える値
	Mx = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 120, normalize_y=True)
	My = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 120, normalize_y=True)
	
	Mx.fit(study_input_list, study_output_list_x)
	My.fit(study_input_list, study_output_list_y)
	
	kkk = Mx.log_marginal_likelihood()
	kkkk = My.log_marginal_likelihood()
	
	params_x = Mx.kernel_.get_params()
	params_y = My.kernel_.get_params()
	
	#print(kkk)
	#print(kkkk)
	#print(params_x)
	#print(params_y)
	
	# save
	joblib.dump(Mx, "modelx.pkl")
	joblib.dump(My, "modely.pkl")

else:
	# load
	Mx = joblib.load("modelx.pkl")
	My = joblib.load("modely.pkl")
	Mx.fit(study_input_list, study_output_list_x)
	My.fit(study_input_list, study_output_list_y)
	
	kkk = Mx.log_marginal_likelihood()
	kkkk = My.log_marginal_likelihood()
	
	params_x = Mx.kernel_.get_params()
	params_y = My.kernel_.get_params()

with open('param.txt', 'w') as f:
	print('Mx : log_marginal_likelihood', file=f)
	print(kkk,params_x, file=f)
	print('My : log_marginal_likelihood', file=f)
	print(kkkk,params_y, file=f)

#-------------------------------------------------------------------

print('bujide')

#simulation
#-------------------------------------------------------------------
#odor_plume_model
CSVV = 'normal1_suko.csv'

#全体座標作成
x=[-100.0,-50.0,0.0,50.0,100.0]
y=[0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0]
xx,yy = np.meshgrid(x,y)
#センサ格子作成
X=[-50.0,0.0,50.0]
Y=[50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0]
XX,YY = np.meshgrid(X,Y)

#----------------------------------------------------------------------------------------------------------------------------
fig = plt.figure()
q = 0
F=0
w=0
success=0
totaltime = 0
pre_c=[]
list=[]
interim_list=[]
timelist= []

CSVV_input = pd.read_csv(CSVV,header=None)
#modelPlume
conc=CSVV_input.values[:, 3:30]
study_time=CSVV_input.values[:, 0]

for inipos in range (-100,101,50):
	SuccessRate=0
	success=0
	totaltime = 0
	avetime = 0
	for ite in range(10):
		#initial_Pos--------------------------------------------
		init_x = np.array([inipos])
		init_y = np.array([500])
		#-------------------------------------------------------
		name1 = '(' + str(init_x[0]) + ',' + str(init_y[0]) + ')' + '_simulation_success_' + str(obst) + '.csv'
		name2 = '(' + str(init_x[0]) + ',' + str(init_y[0]) + ')_' + str(ite) + '_' + str(obst) + '.csv'
		F=0
		print('---------------------------')
		print('simulation_number=' + str(ite))

		for sim in range(len(study_time)):
			if F == 0:
				# plt.clf(): figureをクリア, *plt.cla(): Axesをクリア
				plt.clf()
				#二次元入力、一次元出力_現場位置の濃度用
				input_list = np.array([XX.ravel(), YY.ravel()]).T
				output_list = conc[q,:]
				output_list = output_list.reshape([45,1])

				#これでカーネルを定義します。plume
				kernel = sk_kern.RBF(length_scale=.5)
				#alphaは発散しないように対角行列に加える値
				m = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 80, normalize_y=True)
				m.fit(input_list, output_list)
				pred = np.array([init_x.ravel(), init_y.ravel()]).T
				conc_mean, conc_std = m.predict(pred, return_std=True)
				conc_pred = conc_mean + (conc_std * random.uniform(-1,1))
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

					x_mean, x_std = Mx.predict(pred_x.reshape(1, -1), return_std=True)
					y_mean, y_std = My.predict(pred_y.reshape(1, -1), return_std=True)

					out_x = x_mean + (x_std * random.uniform(-1,1))
					out_y = y_mean + (y_std * random.uniform(-1,1))

					if w == 0:
						init_x = init_x + out_x[0][0]*np.sin(out_y[0][0]*3.14/180)
						init_y = init_y - out_x[0][0]*np.cos(out_y[0][0]*3.14/180)
		
				# print(q)
				print(w)
				print('カイコガPos_x = ' + str(init_x))
				print('カイコガPos_y = ' + str(init_y))
				#CSV吐き出し
				list.extend([study_time[q],init_x[0],init_y[0],out_x[0][0],out_y[0][0],conc_pred[0][0]])
				#CSV吐き出し:time,x,y,conc
				with open(name2, 'a', newline='') as f:
					writer = csv.writer(f)
					# writer.writerow([list[q*4],list[q*4+1],list[q*4+2],list[q*4+3]])
					writer.writerow([list[0],list[1],list[2],list[3],list[4],list[5]])

				print(q)
				q=q+1
				w=0
				conc_pred=0
				out_x=0
				out_y=0
				out_o=0
				list = []

				if sim == (len(study_time)-1):
					q=0
					list = []
					conc_pred=0
					pre_c=[]
					out_x=0
					out_y=0
					out_o=0
					break
					
				#終了条件
				if init_x*init_x + init_y*init_y < 2500:
					# init_x = csv_input.values[0, 9]
					# init_y = csv_input.values[0, 10]
					# init_x = 0
					# init_y = 0
					timelist = CSVV_input.values[q-1, 0]
					totaltime = totaltime + timelist
					q=0
					success = success + 1
					list = []
					pre_c=[]
					conc_pred=0
					out_x=0
					out_y=0
					out_o=0
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
