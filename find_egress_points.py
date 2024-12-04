import numpy as np
import _pickle as pickle
import torch
import time
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os,sys
import torch.multiprocessing as mp
import queue
import copy 



#%% preliminaries


df   = "/home/worklab/Documents/datasets/I24-V/final_detections.npy"



#%% step 1
# get tree for sub-clustering


# load detections
t1 = time.time()
det = np.load(df)
det = torch.from_numpy(det)
det = det[det[:, 0].sort()[1]]

print("Loading data took {:.1f}s".format(time.time() - t1))


# get min and max x and y

minx = torch.min(det[:,1])
maxx = torch.max(det[:,1])
miny = torch.min(det[:,2])
maxy = torch.max(det[:,2])


xsize = int(maxx)
ysize = int(maxy-miny)

im = torch.zeros([xsize+1,ysize+1],dtype = int)

xindex = det[:,1].int()
yindex = (det[:,2]-miny ).int() 

im[xindex,yindex] = 1

# print("Starting iteration")
# for didx,d in enumerate(det):
#     if didx % 100 == 0:
#         print("\r On detection {} of {} -- {:.1f}% done".format(didx,len(det),didx/len(det)*100),end = "\r", flush = True)
        
#     x = int(d[1])
#     y = int(d[2])
#     im[x,y] += 1

im = im.transpose(1,0)
plt.figure(figsize = (30,3))
plt.imshow(im)

zones = { "WB":{
             "source":{
                 "bell":[3300,3400,-120,-65],
                 "hickoryhollow":[6300,6500,-120,-65],
                 "p25":[13300,13500,-100,0],
                 "oldhickory":[18300,18700,-120,-65],
                 "extent":[21900,23000,-100,0]
                 },
             "sink":{
                 "extent":[-1000,0,-100,0],
                 "bell":[4400,4600,-120,-65],
                 "hickoryhollow":[8000,8200,-120,-65],
                 "p25":[13500,13650,-100,0],
                 "oldhickory":[20400,21000,-120,-65],
                 }
             },
          "EB":{
                "source":{
                    "extent":[3500,3600,0,100],
                    "bell":[4200,5000,65,120],
                    "hickoryhollow":[9000,9600,65,120],
                    "p25":[13450,13600,0,100],
                    "oldhickory":[20900,21900,65,120]
                    },
                "sink":{
                    "extent":[21700,23000,0,100],
                    "hickoryhollow1":[6100,6600,70,120],
                    "hickoryhollow2":[7900,8300,65,120],
                    "p25":[13200,13450,0,100],
                    "oldhickory":[18700,19300,65,120]
                    }
                }    
    }