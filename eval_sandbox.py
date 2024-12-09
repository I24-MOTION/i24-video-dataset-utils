import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file  = '/home/worklab/Documents/i24/i24-video-dataset-utils/data/MOTION_v2_post.json'


data = pd.read_json(file)
keys = data.keys()

data = data.values.tolist()

ddata = []
for item in data:
    kv = {}
    for i in range(len(keys)):
        kv[keys[i]] = item[i]
    ddata.append(kv)

durations = []
for item in ddata:
    if item["direction"] == -1:
        dur = item["last_timestamp"] - item["first_timestamp"]
        durations.append(dur)
    
    
print("Average duration over {} trajectories: {:.1f}s".format(len(durations),sum(durations)/len(durations)))
    

plt.figure()
for item in ddata[:20000]:
    if item["direction"] == -1 and item["y_position"][0] > 12 and item["y_position"][0] < 24:
        plt.plot(item["timestamp"],item["x_position"])
plt.show()