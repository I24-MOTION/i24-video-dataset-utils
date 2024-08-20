import numpy as np
import pandas as pd
import cv2
import _pickle as pickle
import torch
import time


def md_iou(a,b,epsilon = 1e-04):
    """
    a,b - [batch_size ,num_anchors, 4]
    """
    
    area_a = (a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])
    area_b = (b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])
    
    minx = torch.max(a[:,:,0], b[:,:,0])
    maxx = torch.min(a[:,:,2], b[:,:,2])
    miny = torch.max(a[:,:,1], b[:,:,1])
    maxy = torch.min(a[:,:,3], b[:,:,3])
    zeros = torch.zeros(minx.shape,dtype=float,device = a.device)
    
    intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
    union = area_a + area_b - intersection + epsilon
    iou = torch.div(intersection,union)
    
    #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
    return iou



#%% what ho!? parameter sittings? More like partyrameter settings!

df   = "/home/worklab/Documents/datasets/I24-V/final_detections.npy"
#df   = "kiou_detections.npy"
gf   = "gap_detections.npy"
gpsf = "/home/worklab/Documents/datasets/I24-V/final_gps.csv"

# select a time and space chunk to work on 
start_time = 0
end_time = 100
start_x = 10000
end_x = 15000
direction = -1

# phase 1 matching overlap parameters
delta = 0.3 #seconds
phi   = 0.5 # required IOU


#%% open gps file, sort and slice
if False:
    t1 = time.time()
    det = np.load(df)
    det = torch.from_numpy(det)
    det = det[det[:, 0].sort()[1]]
    # [time, x,y,l,w,h,class,conf]
    
    time_idxs = torch.where(torch.logical_and(det[:,0] > start_time,det[:,0] < end_time),1,0)
    space_idxs = torch.where(torch.logical_and(det[:,1] > start_x,det[:,1]<end_x),1,0)
    direction_idxs = torch.where(torch.sign(det[:,2]) == direction,1,0)
    keep_idxs = (time_idxs * space_idxs * direction_idxs).nonzero().squeeze()
    
    det = det[keep_idxs,:]
    
    print("Loading data took {:.1f}s".format(time.time() - t1))
    print("For time window [{:.1f}s,{:.1f}s], space window [{:.1f}s,{:.1f}s], --> {} detections".format(start_time,end_time,start_x,end_x,det.shape[0]))
    
    
    #%% Phase 1 tracklet prep
    ### detections have many overlaps - the first phase groups all detections that overlap sufficiently in space, and that create continuous clusters in space (i.e. iou-based tracker)
    t1 = time.time()
    ids = torch.torch.tensor([i for i in range(det.shape[0])]) # to start every detection is in a distinct cluster
    
    for ts in np.arange(start_time,end_time,step = 0.1):
        elapsed = time.time() - t1
        remaining = elapsed/(ts-start_time) * (end_time - ts)
        print("\r Processing time {:.1f}/{:.1f}s     {:.2f}% done, ({:.1f}s elapsed, {:.1f}s remaining.)   ".format(ts,end_time,ts*100/(end_time-start_time),elapsed,remaining),flush = True, end = "\r")
        
        ## grab the set of detections in ts,ts+delta
        ts_idxs = torch.where(torch.logical_and(det[:,0] > ts,det[:,0] < ts+delta),1,0).nonzero().squeeze()
        ts_det = det[ts_idxs,:]
        
        if ts_det.shape[0] == 0:
            continue
        first  = torch.clone(ts_det)
        
        ## convert from state for to state-space box rcs box form
        boxes_new = torch.zeros([first.shape[0],4],device = first.device)
        boxes_new[:,0] = torch.min(torch.stack((first[:,1],first[:,1]+first[:,3]*direction),dim = 1),dim = 1)[0]
        boxes_new[:,2] = torch.max(torch.stack((first[:,1],first[:,1]+first[:,3]*direction),dim = 1),dim = 1)[0]
        boxes_new[:,1] = torch.min(torch.stack((first[:,2]-first[:,4]/2,first[:,2]+first[:,4]/2),dim = 1),dim = 1)[0]
        boxes_new[:,3] = torch.max(torch.stack((first[:,2]-first[:,4]/2,first[:,2]+first[:,4]/2),dim = 1),dim = 1)[0]
        first = boxes_new
        
    
        ## get IOU matrix
        f = first.shape[0]
        first = first.unsqueeze(1).repeat(1,f,1).double()
        ious = md_iou(first,first.transpose(1,0))
        # zero diagonal
        # diag = torch.eye(ious.shape[0])
        # ious = ious - diag
        
        ## get adjacency graph
        adj_graph = torch.where(ious > phi,1,0)
           
        
        ## assign each cluster a unique ID in the cluster ID tensor, or assign it an existing ID if there is already an ID in the cluster ID tensor
        for i in range(len(ts_det)):
            idx = ts_idxs[i] # index into overall det and ids tensors
            
            # get set of ids that match with this detection
            matches = adj_graph[i].nonzero().squeeze(1)
            cluster_idxs = ts_idxs[matches]
            
            if cluster_idxs.shape[0] > 1: # if 1, the only match for the detection is itself
                # get minimum ts_idx
                cluster_ids = ids[cluster_idxs]
                min_id = torch.min(cluster_ids)
    
                # find and replace each other ts_idx
                for id in cluster_ids:
                    ids[ids == id] = min_id
                
    
    ## resulting will be a set of tracklet clusters (clusters of detections)
    count = ids.unique().shape[0]
    out = torch.cat((det,ids.unsqueeze(1)),dim = 1)
    out = out.data.numpy()
    np.save("clustered_{}_{}.npy".format(start_time,end_time),out)
    print("\nFinished clustering detections. Before:{}, After: {}. {:.1f}s elapsed.".format(det.shape[0],count,time.time() - t1))



#%% Phase 2 tracklet to tracklet clustering
if True:
    det = np.load("data/clustered_0_100.npy")
    
    ## ravel tracklets together
    tracklets = {}
    for item in det:
        id = int(item[-1])
        if id not in tracklets.keys():
            tracklets[id] = [item]
        else:
            tracklets[id].append(item)
    
    #flatten into arrays
    for key in tracklets:
        tracklets[key] = np.stack(tracklets[key])
        
    # re-key dictionary to use consecutive 0-start integers so we can use as indices
    
    tracklets_rekeyed = {}
    new_key = 0
    for key in tracklets.keys():
        tracklets_rekeyed[new_key] = tracklets[key]
        new_key += 1
        
    tracklets = tracklets_rekeyed
    print("Raveled tracklets")
    
    
    ## Compute time overlaps for each pair of tracklets
    
    
            
# while tracklets exist that do not have source and sink  ---> This constraint ensures iteration until all tracklets are OD valid
     # valid sources include: present at start, on-ramp, start of space range
     # valid sinks include: present at end, off-ramp, end of space range
     
     # for each pair, compute a set of scores (these can be interatively updated via dynamic programming):
         
     # time score - tracklets are within some time threshold of one another - prevents matches from being made for tracklets that are too far apart - should also prioritize making smaller time-jump matches first
     # proximity score - tracklets are within some space threshold of one another - same as above
         # these first two scores are used to preempt any more time intensive calculations if the tracklets are far apart
     # physical match score - object sizes , classes are the same / similar
     # trajectory score - how well can these tracklets be summarized by a single spline - or perhaps, some simpler polynomial chunks
     # intersection score - this trajectory doesn't intersect any tracklets
     
     
     # select the best-scoring association - flatten these two tracklets into a single tracklet
     # check whether the tracklet has a valid source and sink - if so remove it from further calculation
     
     # now, we need to update the scores for this row/column, remove the old row/column

     # updating the intersection score is non-trivial - technically we need to determine whether every other possible pair would intersect this trajectory
     
     
     
     
     # intersection score - we could use a rasterized roadway map - only one vehicle can occupy each 1 foot by 1 foot grid cell for a timestep - 1 foot grid by 10 Hz = 86 GB ...
     # initially, every trajectory is painted into this

     # alternatively - for every pair where at least one tracklet is physically and temporally proximal
      # determine whether a linear interpolation between that pair intersects this trajectory
      # if we do this, then when two tracklets are combined, their intersection score is an OR operation - if either tracklet A or B could not be paired with tracklet C because of an intersection, then AB cannot be paired with C\
        
     
