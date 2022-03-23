#参考HP
#https://nykergoto.hatenablog.jp/entry/2017/05/29/python%E3%81%A7%E3%82%AC%E3%82%A6%E3%82%B9%E9%81%8E%E7%A8%8B%E5%9B%9E%E5%B8%B0_~_%E3%83%A2%E3%82%B8%E3%83%A5%E3%83%BC%E3%83%AB%E3%81%AE%E6%AF%94%E8%BC%83_~
#https://funatsu-lab.github.io/open-course-ware/machine-learning/gaussian-process/
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
#import GPy
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
CSV = 'Study_gaussian_shigakiOBSTBi.csv'
#-------------------------------------------------------------------

csv_input = pd.read_csv(CSV,header=None)
#study_output:displacement of position (4step - 1step)
study_delta_x=csv_input.values[:, 3]
study_delta_y=csv_input.values[:, 4]
#study_input:current voltage
study_conc1L=csv_input.values[:,5]
#study_input:voltage after 1 step
study_conc2L=csv_input.values[:,7]

#study_input:current voltage
study_conc1R=csv_input.values[:,9]
#study_input:voltage after 1 step
study_conc2R=csv_input.values[:,11]


# if step == 10:
	# study_conc_arma=csv_input.values[:,6]
# elif step == 20:
	# study_conc_arma=csv_input.values[:,7]
# elif step == 30:
	# study_conc_arma=csv_input.values[:,8]
# elif step == 5:
	# study_conc_arma=csv_input.values[:,12]
# # conc=csvv_input.values[:, 3:30]
# study_conc_delta = csv_input.values[:,11]	
# θの値
# study_theta=csv_input.values[:, 13]


#四次元入力、一次元出力_study用
# study_input_list = np.array([study_conc.ravel(), study_conc_arma.ravel(), study_conc_delta.ravel()]).T
# study_input_list = np.array([study_conc_arma.ravel(), study_conc_delta.ravel()]).T
# study_input_list = np.array([study_conc.ravel(), study_conc_arma.ravel()]).T
# study_input_list = np.array([study_conc_arma.ravel()]).T
# study_input_list = np.array([study_conc.ravel()]).T
study_input_list = np.array([study_conc1L.ravel(), study_conc1R.ravel(), study_conc2L.ravel(), study_conc2R.ravel()]).T

study_output_list_x = study_delta_x.reshape(len(csv_input),1)
study_output_list_y = study_delta_y.reshape(len(csv_input),1)


print('bujidesu')

#gaussianのモデル作成
#-------------------------------------------------------------------

if readM == 0:
	#gaussianのモデル作成
	#-------------------------------------------------------------------
	kernel = sk_kern.RBF(length_scale=.5)+sk_kern.WhiteKernel()
	#alphaは発散しないように対角行列に加える値
	Mx = GaussianProcessRegressor(kernel=kernel, alpha = 1e-6, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 120, normalize_y=True)
	My = GaussianProcessRegressor(kernel=kernel, alpha = 1e-6, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 120, normalize_y=True)
	
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
	joblib.dump(Mx, "modelxobs.pkl")
	joblib.dump(My, "modelyobs.pkl")

else:
	# load
	Mx = joblib.load("modelxobs.pkl")
	My = joblib.load("modelyobs.pkl")
	Mx.fit(study_input_list, study_output_list_x)
	My.fit(study_input_list, study_output_list_y)

	kkk = Mx.log_marginal_likelihood()
	kkkk = My.log_marginal_likelihood()
	
	params_x = Mx.kernel_.get_params()
	params_y = My.kernel_.get_params()

with open('paramobs.txt', 'w') as f:
	print('Mx : log_marginal_likelihood', file=f)
	print(kkk,params_x, file=f)
	print('My : log_marginal_likelihood', file=f)
	print(kkkk,params_y, file=f)
#-------------------------------------------------------------------

print('bujide')

#simulation
#-------------------------------------------------------------------
#odor_plume_model
# CSVV = 'J-1_sens_trajectory_merge.csv'

#障害物あるかないか↓ここで設定----------------------------------------------------
#obst = 0
#-------------------------------------------------------------------


#if obst == 0:
#	CSVV = '1Hz_normal_revrev.csv'
#	print("障害物ない")
#elif obst == 1:
#	CSVV = '1Hz_obstacle_revrev.csv'
#	print("障害物あるよ")

#CSVV_input = pd.read_csv(CSVV,header=None)
#modelPlume
#conc=CSVV_input.values[:, 3:48]
#study_time=CSVV_input.values[:, 0]


#全体座標作成
x=[-100.0,-50.0,0.0,50.0,100.0]
y=[0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0]
xx,yy = np.meshgrid(x,y)
#センサ格子作成
X=[-100.0,-50.0,0.0,50.0,100.0]
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
pre_c_L=[]
pre_c_R=[]
list=[]
interim_list=[]
timelist= []

for obst in range (0,2,1):
	if obst == 0:  #1Hz_normal
		CSVV = '1Hz_normal_revrev.csv'
		print("障害物ない")
	elif obst == 1:  #1Hz_obstacle
		CSVV = '1Hz_obstacle_revrev.csv'
		print("障害物あるよ")
	elif obst == 2:  #0.5Hz_normal
		CSVV = '05Hz_normal_revrev.csv'
		print("障害物ない")
	elif obst == 3:  #0.5Hz_obstacle
		CSVV = '05Hz_obstacle_revrev.csv'
		print("障害物あるよ")

	CSVV_input = pd.read_csv(CSVV,header=None)
	#modelPlume
	conc=CSVV_input.values[:, 3:48]
	study_time=CSVV_input.values[:, 0]

	for inipos in range (-50,51,50):
		SuccessRate=0
		success=0
		totaltime = 0
		avetime = 0
		for ite in range(50):
			#initial_Pos--------------------------------------------
			init_x = np.array([inipos])
			init_y = np.array([500])
			#-------------------------------------------------------
			name1 = '(' + str(init_x[0]) + ',' + str(init_y[0]) + ')' + '_simulation_success.csv'
			name2 = '(' + str(init_x[0]) + ',' + str(init_y[0]) + ')_' + str(ite) + '.csv'
			F=0
			print('---------------------------')
			print('simulation_number=' + str(ite))

		
			for sim in range(len(study_time)):
				if F == 0:
					# plt.clf(): figureをクリア, *plt.cla(): Axesをクリア
					plt.clf()
					# global q, csv_input, init_x, init_y, input_dim, conc, XX, YY, CSVV_input, step, Time, pre_c, Mx, My, w, de, study_time,	out_x, out_y, list, interim_list
					#二次元入力、一次元出力_現場位置の濃度用
					input_list = np.array([XX.ravel(), YY.ravel()]).T
					output_list = conc[q,:]
					output_list = output_list.reshape([45,1])
					
					#これでカーネルを定義します。plume
					kernel = sk_kern.RBF(length_scale=.5)
					#alphaは発散しないように対角行列に加える値
					m = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 80, normalize_y=True)
					m.fit(input_list, output_list)
					
					#左右触角(8mm)位置の匂い推定----------------------------------------------
					Left_x = init_x - (8*math.cos(math.radians(0)))
					Left_y = init_y - (8*math.sin(math.radians(0)))
					Right_x = init_x + (8*math.cos(math.radians(0)))
					Right_y = init_y + (8*math.sin(math.radians(0)))
					
					pred_L = np.array([Left_x.ravel(), Left_y.ravel()]).T
					pred_R = np.array([Right_x.ravel(), Right_y.ravel()]).T
					conc_mean_L, conc_std_L = m.predict(pred_L, return_std=True)
					conc_mean_R, conc_std_R = m.predict(pred_R, return_std=True)
					
					conc_pred_L = conc_mean_L + (conc_std_L * random.uniform(-1,1))
					conc_pred_R = conc_mean_R + (conc_std_R * random.uniform(-1,1))
					
					pre_c_L.append(conc_pred_L[0])
					pre_c_R.append(conc_pred_R[0])
					
					#pred = np.array([init_x.ravel(), init_y.ravel()]).T
					#conc_mean, conc_std = m.predict(pred, return_std=True)
					#print('conc_mean = ' + str(conc_mean) + ',    conc_std = ' + str(conc_std))			
					#conc_pred = conc_mean + (conc_std * random.uniform(-1,1))
					# conc_pred = conc_mean

					
					#print('カイコガ位置の推定濃度値fromガウス過程 = ' + str(conc_pred[0]))
					#pre_c.append(conc_pred[0])
					
					# #移動平均濃度forSTEPによる/input
					# if q < step:
						# ave_conc = pre_c[q]
					# else:
						# if step == 10:
							# ave_conc = (pre_c[q] + pre_c[q-1]+pre_c[q-2]+pre_c[q-3]+pre_c[q-4]+pre_c[q-5]+pre_c[q-6]+pre_c[q-7]+pre_c[q-8]+pre_c[q-9])/10
						# elif step == 20:	
							# ave_conc = (pre_c[q] + pre_c[q-1]+pre_c[q-2]+pre_c[q-3]+pre_c[q-4]+pre_c[q-5]+pre_c[q-6]+pre_c[q-7]+pre_c[q-8]+pre_c[q-9]+pre_c[q-10] + pre_c[q-11]+pre_c[q-12]+pre_c[q-13]+pre_c[q-14]+pre_c[q-15]+pre_c[q-16]+pre_c[q-17]+pre_c[q-18]+pre_c[q-19])/20
						# elif step == 30:	
							# ave_conc = (pre_c[q] + pre_c[q-1]+pre_c[q-2]+pre_c[q-3]+pre_c[q-4]+pre_c[q-5]+pre_c[q-6]+pre_c[q-7]+pre_c[q-8]+pre_c[q-9]+pre_c[q-10] + pre_c[q-11]+pre_c[q-12]+pre_c[q-13]+pre_c[q-14]+pre_c[q-15]+pre_c[q-16]+pre_c[q-17]+pre_c[q-18]+pre_c[q-19]+pre_c[q-20] + pre_c[q-21]+pre_c[q-22]+pre_c[q-23]+pre_c[q-24]+pre_c[q-25]+pre_c[q-26]+pre_c[q-27]+pre_c[q-28]+pre_c[q-29])/30	
						# elif step == 5:
							# ave_conc = (pre_c[q] + pre_c[q-1]+pre_c[q-2]+pre_c[q-3]+pre_c[q-4])/5	
					#ΔConc , θ の計算/input
					# if q == 0:
						# de_conc = pre_c[q]
						# theta = 0
					# else:
						# de_conc = (pre_c[q] - pre_c[q-1])/(study_time[q] - study_time[q-1])
						# if out_y[0][0] > 0:
							# if out_x[0][0] > 0:
								# the = math.degrees(math.atan(out_y[0][0]/out_x[0][0])) + 90
								# theta = math.sin(the * math.pi / 180)
							# elif out_x[0][0] < 0:
								# the = math.degrees(math.atan(out_y[0][0]/out_x[0][0])) - 90
								# theta = math.sin(the * math.pi / 180)
						# elif out_y[0][0] < 0:
							# if out_x[0][0] > 0:
								# the = math.degrees(math.atan(out_y[0][0]/out_x[0][0])) + 90
								# theta = math.sin(the * math.pi / 180)
							# elif out_x[0][0] < 0:
								# the = math.degrees(math.atan(out_y[0][0]/out_x[0][0])) - 90
								# theta = math.sin(the * math.pi / 180)
				
		
					#推定ΔxΔyのためのinputデータ/output
					# pred_x = np.array([[conc_pred[0][0], ave_conc, de_conc]])
					# pred_y = np.array([[conc_pred[0][0], ave_conc, de_conc]])
					# pred_x = np.array([[ave_conc, de_conc]])
					# pred_y = np.array([[ave_conc, de_conc]])    
					# pred_x = np.array([conc_pred[0], ave_conc])
					# pred_y = np.array([conc_pred[0], ave_conc])	
					# pred_x = np.array([[conc_pred[0][0]]])
					# pred_y = np.array([[conc_pred[0][0]]])
					# pred_x = np.array([[ave_conc]])
					# pred_y = np.array([[ave_conc]])        	
					if sim > 2:
						pred_x = np.array([pre_c_L[q], pre_c_R[q], pre_c_L[q-2], pre_c_R[q-2]])
						pred_y = np.array([pre_c_L[q], pre_c_R[q], pre_c_L[q-2], pre_c_R[q-2]])
					elif sim == 0:
						pred_x = np.array([pre_c_L[q], pre_c_R[q], [0], [0]])
						pred_y = np.array([pre_c_L[q], pre_c_R[q], [0], [0]])
					elif sim == 1:
						pred_x = np.array([pre_c_L[q], pre_c_R[q], [0], [0]])
						pred_y = np.array([pre_c_L[q], pre_c_R[q], [0], [0]])
					elif sim == 2:
						pred_x = np.array([pre_c_L[q], pre_c_R[q], pre_c_L[q-2], pre_c_R[q-2]])
						pred_y = np.array([pre_c_L[q], pre_c_R[q], pre_c_L[q-2], pre_c_R[q-2]])
					print(pred_x)
					print(pred_y)

					x_mean, x_std = Mx.predict(pred_x.reshape(1, -1), return_std=True)
					y_mean, y_std = My.predict(pred_y.reshape(1, -1), return_std=True)
					#print('x_mean = ' + str(x_mean) + ',    x_std = ' + str(x_std))			
					#print('y_mean = ' + str(y_mean) + ',    y_std = ' + str(y_std))
					out_x = x_mean + (x_std * random.uniform(-1,1))
					out_y = y_mean + (y_std * random.uniform(-1,1))
					# # #↓ ΔxΔy 分散[0],期待値[1]/modelMx,Myを使用
					# out_x = Mx.predict(pred_x)[1]
					# out_y = My.predict(pred_y)[1]
			
					#print('Δx = ' + str(out_x))
					#print('Δy = ' + str(out_y))
					# print('Δθ = ' + str(theta))

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
					if obst == 1:
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
					list.extend([study_time[q],init_x[0],init_y[0],out_x[0][0],out_y[0][0],conc_pred_L[0][0],conc_pred_R[0][0]])
					#CSV吐き出し:time,x,y,conc
					with open(name2, 'a', newline='') as f:
						writer = csv.writer(f)
						# writer.writerow([list[q*4],list[q*4+1],list[q*4+2],list[q*4+3]])	
						writer.writerow([list[0],list[1],list[2],list[3],list[4],list[5],list[6]])		
					# if init_x*init_x + init_y*init_y < 2500:
						# print(q)
					print(q)
					q=q+1
					w=0
					conc_pred=0
					conc_pred_L=0
					conc_pred_R=0
					out_x=0
					out_y=0
					list = []
					# for making time to write data into csv 
					sleep(0.2)
					
					if sim == (len(study_time)-1):
						q=0
						list = []
						conc_pred=0
						conc_pred_L=0
						conc_pred_R=0
						pre_c=[]
						pre_c_L=[]
						pre_c_R=[]
						out_x=0
						out_y=0                
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
						pre_c_L=[]
						pre_c_R=[]
						conc_pred=0
						conc_pred_L=0
						conc_pred_R=0
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
