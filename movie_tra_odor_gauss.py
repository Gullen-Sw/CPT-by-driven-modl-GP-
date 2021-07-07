import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import math
import cv2
import matplotlib.patches as pat
from statistics import mean
from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor
#----------------------------------------------------------------------------------
# #実験2-2(W=200mm)

#全体座標作成
x=list(range(-100, 105, 5))
y=list(range(0, 555 , 5))
xx,yy = np.meshgrid(x,y)
# x=[-100,-50,0,50,100]
# y=[0,50,100,150,200,250,300,350,400,450,500,550]
# xx,yy = np.meshgrid(x,y)

#センサ格子作成
X=[-50,0,50]
Y=[50,100,150,200,250,300,350,400,450]
XX,YY = np.meshgrid(X,Y)

i=0
u=0

#odor map
CSV = 'A-1_conv.csv'
csv_input = pd.read_csv(CSV)

#silkworm moth↓ここで動画にしたい軌跡ファイルを指定
CCSV = '(-100,500)_15.csv'
CCsv_input = pd.read_csv(CCSV)
videoname = '(-100,500).mp4'

#アニメーションの長さ指定。行数
Leng=len(CCsv_input)

fig = plt.figure()

lis = []
listX=[]
listY=[]

listx_L=[]
listy_L=[]
listx_R=[]
listy_R=[]

# sensor2=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

def plot(data):
	# plt.clf(): figureをクリア, *plt.cla(): Axesをクリア
	plt.clf()
	global i, csv_input, ccsv_input, lis, u, listX, listY, listx_L, listy_L, listx_R, listy_R
	
	sss = [50,100,150,200,250,300,350,400,450]
	ssss = [-50, 0, 50]
	
	#from csv_input--------------------------------------
	#センサ格子点の高さの値
	sensor=csv_input.values[i, 1:46]
	#吐出表示
	duration=csv_input.values[i, 46]

	#from CCsv_input--------------------------------------	
	#カイコガの座標
	trajectory= CCsv_input.values[i, 1:3]
	# #カイコガの姿勢
	# dig= CCsv_input.values[i, 5]
	#時間表示
	time= CCsv_input.values[i, 0]


	print('time=' + str(time))
	print(trajectory[0])
	print(trajectory[1])
	print(duration)
	print(sensor)

	input_list = np.array([XX.ravel(), YY.ravel()]).T
	output_list = sensor
	output_list = output_list.reshape([27,1])
	
	#これでカーネルを定義します。plume
	kernel = sk_kern.RBF(length_scale=.5)
	#alphaは発散しないように対角行列に加える値
	m = GaussianProcessRegressor(kernel=kernel, alpha = 1e-5, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer = 70, normalize_y=True)
	m.fit(input_list, output_list)
#Gaussian
	for yyy in range(0, 555 , 5):
		for xxx in range(-100, 105, 5):
			if yyy in sss:
				if xxx in ssss:
					lis.append(output_list[u])
					u=u+1
				else:
					xxx = np.array([xxx])
					yyy = np.array([yyy])
					pred = np.array([xxx.ravel(), yyy.ravel()]).T
					conc_mean, conc_std = m.predict(pred, return_std=True)
					conc_pred = conc_mean
					if conc_pred[0][0]<0:
						conc_pred[0][0]=0
					lis.append(conc_pred[0][0])


			else:
				xxx = np.array([xxx])
				yyy = np.array([yyy])
				pred = np.array([xxx.ravel(), yyy.ravel()]).T
				conc_mean, conc_std = m.predict(pred, return_std=True)
				conc_pred = conc_mean
				if conc_pred[0][0]<0:
					conc_pred[0][0]=0
				lis.append(conc_pred[0][0])

	liss = np.array(lis)
	print(liss)
	sensor_matrix=liss.reshape(111,41)

#plt.contourf等高線作る。全体座標。np.linspaceでカラーバーのスケール設定と刻み幅の指定。	
	v = np.linspace(0, 3, 41, endpoint=True)
	plt.contourf(xx, yy, sensor_matrix ,v ,cmap='viridis')
	plt.xlabel('x [mm]')
	plt.ylabel('y [mm]')

#センサ格子カラーバー幅指定。
	L=plt.colorbar()
	L.set_label('Voltage [V]')

# plt.xlimでグラフの幅指定
	plt.xlim([-100,100])
	plt.ylim([0,550])
# plt.scatterは散布図作成。今回はセンサ格子の位置を出現。sがプロットの大きさ指定。alphaが透過性。
	plt.scatter(XX, YY, s=50, edgecolors="black",facecolor='None',marker='s',alpha=0.5)

# 障害物出現-----------------------------------------------------------------
	# ww = np.arange(-math.pi , math.pi, 0.01)
	# for ww in ww:
		# plt.scatter(20*math.cos(ww) , 20*math.sin(ww)+250 , c='black', s = 0.1)
#-------------------------------------------------------------------------

# goal半径出現
	w = np.arange(-math.pi , math.pi, 0.01)
	for w in w:
		plt.scatter(50*math.cos(w) , 50*math.sin(w) , c='black', s = 0.1)
# startPos出現
	plt.scatter(0 , 500 , c='black', s = 10)
	
# カイコガ位置出現
	plt.scatter(trajectory[0] , trajectory[1] ,edgecolors="blue", marker="o", c='orange', s = 50)

# # カイコガ姿勢出現

	# listX.extend([trajectory[0],trajectory[0]+13*math.sin(math.radians(dig))])
	# listY.extend([trajectory[1],trajectory[1]-13*math.cos(math.radians(dig))])
	
    # #(x1,y1)から(x2,y2)まで直線を引く
	# plt.plot(listX,listY,c='black',linewidth=1.4)
	# listX=[]
	# listY=[]

# # カイコガ触角出現

	# listx_L.extend([trajectory[0],trajectory[0]- 8*math.cos(math.radians(dig))])
	# listy_L.extend([trajectory[1],trajectory[1]- 8*math.sin(math.radians(dig))])
	# listx_R.extend([trajectory[0],trajectory[0]+ 8*math.cos(math.radians(dig))])
	# listy_R.extend([trajectory[1],trajectory[1]+ 8*math.sin(math.radians(dig))])
	# #(x1,y1)から(x2,y2)まで直線を引く
	# plt.plot(listx_L,listy_L,c='black',linewidth=0.8)
	# plt.plot(listx_R,listy_R,c='black',linewidth=0.8)
	# listx_L=[]
	# listy_L=[]
	# listx_R=[]
	# listy_R=[]


# 時間出現(x,y,表示文字)
	plt.text(-100,600,"T = " + str(time) + " s")

# 吐出タイミング出現(x,y,表示文字)
	if duration==1:
		odor="On"
	else:
		odor="Off"
	plt.text(-100,575,"EmissionSwitch : " + str(odor))

# アスペクト比設定
	plt.axes().set_aspect('equal')
	
# 変数初期化	
	i=i+1
	lis=[]
	liss=0
	u=0


ani=animation.FuncAnimation(fig, plot, frames=Leng)
ani.save(videoname, writer="ffmpeg", fps=30)
#plt.show()


