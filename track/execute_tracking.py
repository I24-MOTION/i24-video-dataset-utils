import numpy as np
import os
import torch
import time
import _pickle as pickle 

from tracker import SmartTracker,HungarianIOUAssociator,EuclideanAssociator,ByteIOUAssociator, ByteEucAssociator
from trackstate import TrackierState
import bbox
from i24_rcs import I24_RCS

#%% Set all yer parameters 

os.environ["USER_CONFIG_DIRECTORY"] = "/home/worklab/Documents/i24/fast-trajectory-annotator/tracking/config"

stride = 1/10
min_conf = 0.05


save_name = "BYTE_EUC_10Hz_2.cpkl"

#%% Import detections
detections = "/home/worklab/Desktop/64cbe0ff7ef4a8ecc60a2597_sorted.npy"
detections = np.load(detections)
detections = torch.from_numpy(detections)


#%% Initialize stuff or reload TrackState if applicable
hg_file = "/home/worklab/Documents/i24/i24_track/data/homography/CIRCLES_20_Wednesday.cpkl"
hg      = I24_RCS(save_file = hg_file,downsample = 2)
tracker = SmartTracker()
tstate  = TrackierState()
ass     = ByteEucAssociator() #ByteIOUAssociator() # hehehe I am 27 years old  
term_objects = []


tracker.fsld_max = int(1/stride) *2
min_track_length = int(1/stride)

ts_start = detections[0,0]
ts_end   = detections[-1,0]


last_didx = 0
tracking_start_time = time.time()

#%% main loop
t1 = ts_start
for tidx,t2 in enumerate(np.arange(ts_start+stride,ts_end,step = stride)):
    active = len(tstate)
    terminated = len(term_objects)
    sec    = str(int((t2-ts_start)    % 60)).zfill(2)
    minute = str(int((t2 - ts_start) // 60)).zfill(2)
    
    elapsed = time.time() - tracking_start_time
    eta = elapsed * (ts_end-ts_start)/(t2 - ts_start) -elapsed
    eta_sec = str(int(eta)%60).zfill(2)
    eta_min = str(int(eta)//60)
    
    print("\rChunk {} --- {}:{} processed so far --- {} active, {} terminated  --- ETA {}:{}".format(tidx,minute,sec,active,terminated,eta_min,eta_sec),end = "\r", flush = True)
    if tidx%1000 == 0:
        print("\nSaved copy of data")
        
        with open(save_name,"wb") as f:
            pickle.dump([ts_start,term_objects],f)

    # Bite off a nice chewable chunk of the detection filet
    this_didx = last_didx
    while detections[this_didx,0] < t2:
        this_didx += 1
    det = detections[last_didx:this_didx,:]
    
    
    #### chew and swallow it into into tracklets

    # filter by confidence
    keep = torch.where(det[:,-1]> min_conf,1,0,).nonzero().squeeze(-1)
    det = det[keep]

    # non-maximal supression
    dbox = torch.cat((det[:,1:6],torch.sign(det[:,2]).unsqueeze(1)),dim = 1)
    keep = bbox.state_nms(dbox.float(),det[:,-1].float(),threshold = 0.1)
    det = det[keep,:]

    
    scores = det[:,-1]
    times = det[:,0] - ts_start
    dbox  = torch.cat((det[:,1:6],torch.sign(det[:,2]).unsqueeze(1)),dim = 1)
    classes = det[:,6]
    
    # prediction - a bit of timestamp intricacy here probably
    obj_times = torch.ones([len(tstate)]) * times.mean()
    obj_ids, priors, _ = tracker.preprocess(tstate, obj_times)
    
    
    
    
    # association
    associations = ass(obj_ids,priors,dbox,hg)
    
    # pass objects to tracker
    terminated_objects,COD = tracker.postprocess(dbox,times, classes, scores, associations, tstate, hg = hg,measurement_idx =0)
    
    # Do some book-keeping
    t1 = t2
    last_didx = this_didx
    
    for t in terminated_objects.keys():
        obj = terminated_objects[t]
        if len(obj[0]) > min_track_length:
           data = torch.stack([obj[0][i][1] for i in range(len(obj[0]))])
           times = torch.stack([obj[0][i][0]+ts_start for i in range(len(obj[0]))]).unsqueeze(1)
           confs = torch.stack(obj[-1]).unsqueeze(1)
           data = torch.cat((times,data,confs),dim = 1)
           cls = np.bincount(obj[1].astype(int)).argmax()
           term_objects.append((data,cls))
           
   
    
    
# Flush
terminated_objects,COD  = tracker.flush(tstate)
if len(obj[0]) > min_track_length:
   data = torch.stack([obj[0][i][1] for i in range(len(obj[0]))])
   times = torch.stack([obj[0][i][0]+ts_start for i in range(len(obj[0]))]).unsqueeze(1)
   confs = torch.stack(obj[-1]).unsqueeze(1)
   data = torch.cat((times,data,confs),dim = 1)
   cls = np.bincount(obj[1].astype(int)).argmax()
   term_objects.append((data,cls))   
   
with open(save_name,"wb") as f:
    pickle.dump([ts_start,term_objects],f)