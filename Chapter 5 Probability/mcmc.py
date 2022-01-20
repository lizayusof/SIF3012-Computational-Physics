#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def func(x,a,b):
    if (a<x<b):
        return 1.0/(1.0+x**2)
    else:
        return 0

def areaMCMC(a,b,hops=10_000):
    current = np.random.uniform(a,b)
    samples = []
    for i in range(hops):
        samples.append(current)
        disturbance = np.random.uniform(-0.25,0.25)
        movement = current + disturbance
        
        current_lik = func(x=current,a=a,b=b)
        movement_lik = func(x=movement,a=a,b=b)
        ratio = movement_lik/current_lik
        
        event = np.random.uniform(0,1)
        if event <= ratio:
            current = movement
    return samples

samples = areaMCMC(a=0,b=3)

answer = sum(samples)/len(samples)
print(answer)

#import matplotlib,pyplot as plt
plt.hist(samples,density=True)