import math, time, sys, random, re
import pandas as pd, numpy as np
from numpy import genfromtxt

startTime, alpha, invals, i_use, answers, errsum, minerrsum, layercts = time.time(), 0.1, [], [], [], 0.01, 0.02, [784, 256, 128, 64, 10]

for a in genfromtxt('train.csv', delimiter=',')[1:]:
    invals.append(list([int(b)/255 for b in a[2:]]))
    i_use.append(list([int(b)/255 for b in a[2:]]))
    answers.append(int(a[1]))

weights = [[[random.uniform(-1,1) for a in range(layercts[0])] for b in range(layercts[1])], [[random.uniform(-1,1) for a in range(layercts[1])] for a in range(layercts[2])], [[random.uniform(-1,1) for a in range(layercts[2])] for a in range(layercts[3])], [[random.uniform(-1,1) for a in range(layercts[3])] for a in range(layercts[4])]]
w_use = []
for data in weights: 
    w_use.append( [] )
    for elem in data:
        w_use[-1].append( [num for num in elem] )

def backprop(err, err_weights, xvals, err_nodes, weights):
    err_nodes[-1][-1] = err
    answer = 0.0
    for data in weights: 
        err_weights.append( [] )
        for elem in data:
            err_weights[-1].append( [0] * len(elem) )
    for i in range(len(err_nodes)):
        ii = -i-1
        for j in range(len(err_nodes[ii])):
            if not (i or j): continue
            err_nodes[ii][j] = xvals[ii][j] * (1.0-xvals[ii][j]) * err_nodes[ii+1][0] * weights[ii+1][0][j]
    for i in range(len(weights)):
        ii = -i-1
        for j in range(len(weights[ii])):
            for k in range(len(weights[ii][j])):
                err_weights[ii][j][k] = err_nodes[ii][j] * xvals[ii-1][k]
    return err_weights

def apply():
    xvals, yvals, final, ans = [], [], [], []
    while i_use:
        temp = i_use.pop(0)
        xvals.append( [] )
        yvals.append( [] )
        while w_use:
            pound, newtemp, x = w_use.pop(0), [], 0.0
            xvals[len(invals)-len(i_use)-1].append( [val for val in temp] )
            for y in range(len(pound)):
                x = 0
                # print(temp)
                # print(pound[y])
                # print(len(temp))
                for z in range(len(temp)):
                    x += temp[z]*pound[y][z]
                newtemp.append(x)
            yvals[len(invals)-len(i_use)-1].append( [val for val in newtemp] )
            if w_use:
                for val in newtemp: temp.append(1.0/(1.0+math.exp(-val)))
            else: temp = [val for val in newtemp]
        xvals[len(invals)-len(i_use)-1].append( [val for val in temp] )
        err = answers[len(invals)-len(i_use)-1]-temp[0]
        final.append(err)
        ans.append(temp[0])
        change = backprop(err, [], xvals[-1], [[0 for a in b] for b in xvals[-1][1:]], weights)
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    weights[i][j][k] += alpha * change[i][j][k]
        for data in weights: w_use.append( [elem for elem in data] )
    xvals, yvals = [], []
    for data in invals: i_use.append( [elem for elem in data] )
    return final, ans

bestweights = []
print("\nLayer cts:", layercts)
for data in weights: bestweights.append( [elem for elem in data] )
while time.time()-startTime < 99.5:
    err, ans = apply()
    errsum = sum(abs(sel) for sel in err)
    if errsum < minerrsum:
        minerrsum = errsum
        bestweights = []
        for data in weights: bestweights.append( [elem for elem in data] )

invals, i_use = [], []
for a in genfromtxt('train.csv', delimiter=',')[1:]:
    invals.append(list([int(b)/255 for b in a[1:]]))
    i_use.append(list([int(b)/255 for b in a[1:]]))

xvals, yvals, final = [], [], []
n = 0

submission = np.array([['id', 'label']])
while i_use:
    temp = i_use.pop(0)
    xvals.append( [] )
    yvals.append( [] )
    while w_use:
        pound, newtemp, x = w_use.pop(0), [], 0.0
        xvals[len(invals)-len(i_use)-1].append( [val for val in temp] )
        for y in range(len(pound)):
            x = 0
            for z in range(len(temp)): x += temp[z]*pound[y][z]
            newtemp.append(x)
        yvals[len(invals)-len(i_use)-1].append( [val for val in newtemp] )
        if w_use:
            for val in newtemp: temp.append(1.0/(1.0+math.exp(-val)))
        else: temp = [val for val in newtemp]
    xvals[len(invals)-len(i_use)-1].append( [val for val in temp] )
    numpy.append(submission, [n, temp[0]])
    n += 1
    
np.savetxt("submission.csv", submission, delimiter=",")