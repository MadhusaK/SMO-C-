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
d_reclassData = np.loadtxt('reclassData', delimiter = ",")


###con data
d_conData = np.loadtxt('conData', delimiter = ",")
d_conClass = np.loadtxt('conClass', delimiter = ",")

### Cluster data

d_clusterData = np.loadtxt('clusterData', delimiter = ",")
d_clusterClass =  np.loadtxt('clusterClass', delimiter = ",")



pyplot.figure(0)
pyplot.hold(True)


for i in range(np.shape(d_clusterClass)[0]):
    if (d_clusterClass[i] == 1):
        pyplot.plot(d_clusterData[0][i], d_clusterData[1][i], 'ro')
    else:
        pyplot.plot(d_clusterData[0][i], d_clusterData[1][i], 'bo')


pyplot.figure(1)
pyplot.hold(True)


for i in range(np.shape(d_clusterClass)[0]):
    if (d_reclassData[i] == 1):
        pyplot.plot(d_clusterData[0][i], d_clusterData[1][i], 'ro')
    else:
        pyplot.plot(d_clusterData[0][i], d_clusterData[1][i], 'bo')