# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:56:52 2016

@author: madhusa
"""

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib as mpl

d_testData = np.loadtxt('testData', delimiter = ",")
d_classData = np.loadtxt('testClass', delimiter = ",")

pyplot.figure(0)
pyplot.hold(True)


for i in range(np.shape(d_classData)[0]):
    if (d_classData[i] == 1):
        pyplot.plot(d_testData[0][i], d_testData[1][i], 'ro')
    else:
        pyplot.plot(d_testData[0][i], d_testData[1][i], 'bo')
