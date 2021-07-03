# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:54:25 2021

@author: LEE
"""

import pandas as pd

#data sheet
CSV = 'A-1_TrimDLC_resnet50_Obstacle_blackJun5shuffle1_1030000.csv'

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
                    
while csv_input.values[i, 0]!=None:
#head x 2step & delta_x 
   temp_h_x_1 = (csv_input.values[i, 1] - init_x)/rate
   temp_h_x_2 = (csv_input.values[i+1, 1] - init_x)/rate
   h_study_delta_x=temp_h_x_2 - temp_h_x_1
 
#head y 2step & delta_y
   temp_h_y_1 = (csv_input.values[i, 2] - init_y)/rate
   temp_h_y_2 = (csv_input.values[i+1, 2] - init_y)/rate
   h_study_delta_y=temp_h_y_2 - temp_h_y_1

   csv_input.loc[i, "head_x_t(mm)"] = temp_h_x_1
   csv_input.loc[i, "head_x_t+1(mm)"] = temp_h_x_2
   csv_input.loc[i, "head_delta_x(mm/s)"] = h_study_delta_x
   csv_input.loc[i, "head_y_t(mm)"] = temp_h_y_1
   csv_input.loc[i, "head_y_t+1(mm)"] = temp_h_y_2
   csv_input.loc[i, "head_delta_y(mm/s)"] = h_study_delta_y
   
#tail x 2step & delta_x 
   temp_t_x_1 = (csv_input.values[i, 4] - init_x)/rate
   temp_t_x_2 = (csv_input.values[i+1, 4] - init_x)/rate
   t_study_delta_x=temp_t_x_2 - temp_t_x_1
  
#tail y 2step & delta_y
   temp_t_y_1 = (csv_input.values[i, 5] - init_y)/rate
   temp_t_y_2 = (csv_input.values[i+1, 5] - init_y)/rate
   t_study_delta_y=temp_t_y_2 - temp_t_y_1
   
   csv_input.loc[i, "tail_x_t(mm)"] = temp_t_x_1
   csv_input.loc[i, "tail_x_t+1(mm)"] = temp_t_x_2
   csv_input.loc[i, "tail_delta_x(mm/s)"] = t_study_delta_x
   csv_input.loc[i, "tail_y_t(mm)"] = temp_t_y_1
   csv_input.loc[i, "tail_y_t+1(mm)"] = temp_t_y_2
   csv_input.loc[i, "tail_delta_y(mm/s)"] = t_study_delta_y

   i = i+1
   if i == max(r):
       break

#save File
csv_input.to_csv('Test.csv', index = False, encoding = 'cp949') 


