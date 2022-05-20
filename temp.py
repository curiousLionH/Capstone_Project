# from sympy import Symbol, solve
# import numpy as np
# import math as m
# L = 100
# l1 = 50
# ls = 20
# m = (l1+ls)/5
# Z = 110     # 눈Z
# X = 50      # 눈X
# theta = Symbol('theta')
# print("")
# eq1 = np.arctan((L-m*np.cos(theta)) / (m*np.sin(theta))) + np.arctan((Z-m*np.cos(theta)) / (X + m*np.sin(theta))) - 2*theta
# ans = solve(eq1)
# print(ans)
# # a = m.atan2(-0.5,-0.5)
# # print(a)

import cv2 as cv 

def testDevice(source):
   cap = cv.VideoCapture(source) 
   if cap is None or not cap.isOpened():
       print('Warning: unable to open video source: ', source)

for i in range(8):
    testDevice(i) 