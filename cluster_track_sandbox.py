import numpy as np
import pandas as pd
import cv2
import _pickle as pickle
import torch
import time
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os




#%% what ho!? parameter settings? More like partyrameter settings!

df   = "/home/worklab/Documents/datasets/I24-V/final_detections.npy"
#df   = "kiou_detections.npy"
gf   = "gap_detections.npy"
gpsf = "/home/worklab/Documents/datasets/I24-V/final_gps.csv"

# select a time and space chunk to work on 
start_time = 0
end_time = 60
start_x = 9000
end_x  = 12000
direction = -1

# phase 1 matching overlap parameters
delta = 0.3 #seconds
phi   = 0.4 # required IOU


# phase 2 overlap parameters
t_thresholds = [1,2,2,2,2,3,4,5,5,5,6,6,6,6,6,6,7,7]                 # tracklets beyond this duration apart from one another are not considered
cutoff_dists = [3,5,7,8,10,10,10,10,12,15,15,17,18,20,22,22,22,30]          # tracklets with a minimum scored match of > _ ft apart are not matched
x_thresholds = [400 for _ in t_thresholds]             # tracklets beyond this distance apart from one another are not considered
y_thresholds = [2,5,5,5,6,6,7,8,8,8,8,8,8,8,8,8,8,8]                 # tracklets beyond this distance apart from one another are not considered
reg_keeps    = [20 for _ in t_thresholds]              # the last _ points are used to fit a linear regression and the first _ points from the next tracklet are scored
min_regression_lengths = [30 for _ in t_thresholds]    # tracklets less than _ in length are not used to fit a linear regression
min_duration = 1
t_buffer = 0

x_margin = 50
t_margin = 2
big_number = 10000

lane = 1


time_chunk = 60
space_chunk = 2000

time_chunk = 60
space_chunk = 3000
# x_threshold           = x_thresholds[0]                  # tracklets beyond this distance apart from one another are not considered
# y_threshold           = y_threshold = y_thresholds[0]    # tracklets beyond this distance apart from one another are not considered
# t_threshold           = t_thresholds[0]                  # tracklets beyond this duration apart from one another are not considered
# reg_keep              = reg_keeps[0]                     # the last _ points are used to fit a linear regression and the first _ points from the next tracklet are scored
# min_regression_length = min_regression_lengths[0]        # tracklets less than _ in length are not used to fit a linear regression
# cutoff_dist           = cutoff_dists[0]    


def get_msf_points(tracklet,msf_dict,extra_t = 10,freq = 0.1,direction = -1):
    """
    Returns virtual tracklet points using msf for extra_t seconds after end of tracklet, sampled at freq Hz
    
    msf_dict = {
        "msf":msf_smoothed,
        "start_x":start_x,
        "end_x":end_x,
        "start_t":start_t,
        "end_t":end_t,
        "t_grid":t_grid,
        "x_grid":x_grid,
        }
    
    """
    msf = msf_dict["msf"]
    msf = torch.nan_to_num(msf,0)

    # get start x and t
    start_x = tracklet[-1,1]
    start_t = tracklet[-1,0]
    
    t = start_t
    x = start_x
    lane = int((tracklet[-1,2] // -12).item())
    msf = msf[lane]
    
    virtual = [[t,x]]
    while t < start_t + extra_t and t < msf_dict["end_t"]:
        
        # get msf bin
        t_bin = int((t-msf_dict["start_t"]) // msf_dict["t_grid"])
        if t_bin >= msf.shape[1]: break
        
        x_bin = int((x-msf_dict["start_x"]) // msf_dict["x_grid"])
        if x_bin >= msf.shape[0]: break 
        
        speed = msf[x_bin,t_bin]
        x = x + (direction * speed*freq)
        t = t + freq
        virtual.append([t,x])
    
    virtual = torch.tensor(virtual)
    return virtual


def plot_tracklet(tracklets,i = -1,TEXT = True, lane = 1, virtual = None):
    plt.close()
    plt.figure(figsize = (20,15))

    for tidx,t in enumerate(tracklets):
        if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
            plt.plot(t[:,0],t[:,1],c = colors[tidx])
            if TEXT: plt.text(t[0,0],t[0,1],tidx)
    for tidx,t in enumerate(tracklets_complete):
        if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
            plt.plot(t[:,0],t[:,1],c = "b")
    plt.title("Post clustering lane {} after pass {}".format(lane,iteration))
    plt.show()
    if i == -1: return
    
    
    
    
    # fit linear regressor
    t1  = tracklets[i][-reg_keep:,0].unsqueeze(1)
    y1  = tracklets[i][-reg_keep:,1:3]
    reg = LinearRegression().fit(t1,y1)

    
    # fit linear regressor
    t1b  = tracklets[i][:reg_keep,0].unsqueeze(1)
    y1b  = tracklets[i][:reg_keep,1:3]
    regb = LinearRegression().fit(t1b,y1b)

    plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
    
    if True or virtual is None:
        #plot regression line
        try: t_trend = np.array([[tracklets[i][-reg_keep,0]],[tracklets[i][-1,0]+t_threshold]])
        except: t_trend = np.array([[tracklets[i][-1,0]],[tracklets[i][-1,0]+t_threshold]])
        y_trend = reg.predict(t_trend)
        plt.plot(t_trend,y_trend[:,0],":",c = "k")
        
        #plot backward regression line
        try: t_trend = np.array([[tracklets[i][reg_keep,0]],[tracklets[i][0,0]-t_threshold]])
        except: t_trend = np.array([[tracklets[i][0,0]],[tracklets[i][0,0]-t_threshold]])
        y_trend = regb.predict(t_trend)
        plt.plot(t_trend,y_trend[:,0],"--",c = "k")
    
    for j in range(len(tracklets)):
        if intersection[i,j] == 1:
            
            # get first bit of data
            t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
            y2  = tracklets[j][:reg_keep,1:3]
            
            pred = reg.predict(t2)
            diff = np.abs(pred - y2.data.numpy())
            
            mdx = diff[:,0].mean()
            mdy = diff[:,1].mean()
            align_x[i,j] = mdx
            align_y[i,j] = mdy
            
            
   
            
            plt.scatter(tracklets[j][:,0],tracklets[j][:,1], c = "r")
            if TEXT: plt.text(tracklets[j][0,0],tracklets[j][0,1]+5,"{:.1f}ft".format(align_x[i,j].item()),color = (0.5,0.5,0.5))

            
        if intersection[j,i] == 1:
            t2b = tracklets[j][-reg_keep:,0].unsqueeze(1)
            y2b = tracklets[j][-reg_keep:,1:3]
            
            pred = regb.predict(t2b)
            diff = np.abs(pred - y2b.data.numpy())
            
            mdx = diff[:,0].mean()
            mdy = diff[:,1].mean()
            align_xb[j,i] = mdx
            align_yb[j,i] = mdy
        
            plt.scatter(tracklets[j][:,0],tracklets[j][:,1], color = (0.8,0.8,0))
            if TEXT: plt.text(tracklets[j][-1,0],tracklets[j][-1,1]+5,"{:.1f}ft".format(align_xb[j,i].item()),color = (0.5,0.5,0.5))


            
    # find min alignment error
    min_idx = torch.argmin(align_x[i]**2 + align_y[i]**2)
    min_dist = torch.sqrt(align_x[i,min_idx]**2 + align_y[i,min_idx]**2)
    if min_dist < cutoff_dist:
        plt.scatter(tracklets[min_idx][:,0],tracklets[min_idx][:,1], c = "b")
        plt.title("Minimum mean distance: {:.1f}ft".format(min_dist))
    
    # min backwards match
    min_idx = torch.argmin(align_xb[:,i]**2 + align_yb[:,i]**2)
    min_dist2 = torch.sqrt(align_xb[min_idx,i]**2 + align_yb[min_idx,i]**2)
    if min_dist2 < cutoff_dist:
        plt.scatter(tracklets[min_idx][:,0],tracklets[min_idx][:,1], color = (0,0.7,0.7))
        plt.title("Minimum mean distance: {:.1f}ft,{:.1f}ft".format(min_dist,min_dist2))
        
    if virtual is not None:
        plt.plot(virtual[:,0],virtual[:,1], "-.", color = (0.1,0.7,0.2))
    
    plt.show()

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def generate_msf(tracklets,start_time,end_time,start_x,end_x,grid_t = 2,grid_x = 50,kwidth = 5,lane = 1, SHOW = False):
    
    # create holder matrix
    t_range = torch.arange(start_time,end_time,grid_t)
    t_range2 = torch.cat((t_range[1:],torch.tensor(end_time).unsqueeze(0)))
    x_range = torch.arange(start_x,end_x,grid_x)
    x_range2 = torch.cat((x_range[1:],torch.tensor(end_x).unsqueeze(0)))

    distance = torch.zeros(x_range.shape[0],t_range.shape[0],7)
    time     = torch.zeros(x_range.shape[0],t_range.shape[0],7) 
    
    for tdx,tracklet in enumerate( tracklets):
        if tdx%100 == 0: print("Adding tracklet {} to mean speed field".format(tdx))
        # get the points that belong in each time and space bin
        # yep, you guessed it, it's gonna be another 3-4D tensor op
        # there are p points. we need to determine whether each point is > t1 and < t2
        p  = tracklet.shape[0]
        tra_x = tracklet[:,1].unsqueeze(1).expand(p,x_range.shape[0])
        tra_t = tracklet[:,0].unsqueeze(1).expand(p,t_range.shape[0])
        
        x1 = x_range.unsqueeze(0).expand(p,x_range.shape[0])
        x2 = x_range2.unsqueeze(0).expand(p,x_range.shape[0])
        
        xbin = torch.where(torch.logical_and(tra_x > x1,tra_x < x2),1,0).nonzero()
        
        t1 = t_range.unsqueeze(0).expand(p,t_range.shape[0])
        t2 = t_range2.unsqueeze(0).expand(p,t_range.shape[0])
        tbin = torch.where(torch.logical_and(tra_t > t1,tra_t < t2),1,0).nonzero()
        
        
            
        minmax = {}
        
        for i in range(p):
            try:
                m = torch.where(tbin[:,0] == i,1,0).nonzero().squeeze().item()
                n = torch.where(xbin[:,0] == i,1,0).nonzero().squeeze().item()
                lane  = int((tracklet[i,2] // -12).item())
            except RuntimeError: continue # occurs when the point falls outside of all bin ranges
            
            if "{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane) not in minmax.keys():
                minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)] = {"xmin":tracklet[i,1],
                                                           "xmax":tracklet[i,1],
                                                           "tmin":tracklet[i,0],
                                                           "tmax":tracklet[i,0]}
            else:
                minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["xmin"] = min(tracklet[i,1],minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["xmin"])        
                minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["xmax"] = max(tracklet[i,1],minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["xmax"])   
                minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["tmin"] = min(tracklet[i,0],minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["tmin"])        
                minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["tmax"] = max(tracklet[i,0],minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["tmax"])   
        
        for key in minmax.keys():
            xidx = int(key.split(":")[0])
            tidx = int(key.split(":")[1])
            lidx = min(6,int(key.split(":")[2])) # make sure no out-of-bounds data
            
            t_elapsed = minmax[key]["tmax"] - minmax[key]["tmin"]
            if t_elapsed > 0.2:
                x_elapsed = minmax[key]["xmax"] - minmax[key]["xmin"]
            
                distance[xidx,tidx,lidx] += x_elapsed
                time[xidx,tidx,lidx] += t_elapsed
    
    all_msf_unsmooth = []    
    all_msf= []
    for lane in range(0,7):
        msf = distance[:,:,lane]/(time[:,:,lane]+0.01)
        all_msf_unsmooth.append(msf)

        if SHOW:
            p =plt.imshow(msf)
            plt.colorbar(p)
            plt.title("Mean Speed Field with missing values")
            plt.show()
        
        ## interpolate missing values 
        mask = torch.where(msf > 0,1,0)
        
    
        pad = int((kwidth -1 )//2)
        # convolve a kernel
        kernel = torch.from_numpy(gkern(kwidth,1.5)).float()
        kernel_sum = kernel.sum()
        counter = (kernel*0+1).float() # all 1s
        
        conv1 = torch.nn.functional.conv2d(msf.unsqueeze(0).unsqueeze(0),kernel.unsqueeze(0).unsqueeze(0),padding = pad).squeeze(0).squeeze(0)
        conv2 = torch.nn.functional.conv2d(mask.float().unsqueeze(0).unsqueeze(0),kernel.unsqueeze(0).unsqueeze(0),padding = pad).squeeze(0).squeeze(0) / kernel_sum  # what proportion of kernel weights were used?
        
        msf_smoothed = conv1/conv2
        all_msf.append(msf_smoothed)
        if False:
            msf_smoothed = msf_smoothed * (1-mask) + msf*mask
        
        if SHOW:
            p =plt.imshow(msf_smoothed)
            plt.colorbar(p)
            plt.title("Mean Speed Field after interpolation / smoothing")
            plt.show()
    
    # weight the result such that the kernel elements corresponding to non-zero values have total weight = 1
    
    # mask the original matrix and only replace zero values
    msf_dict = {
        "msf":torch.stack(all_msf),
        "msf_raw":torch.stack(all_msf_unsmooth),
        "start_x":start_x,
        "end_x":end_x,
        "start_t":start_time,
        "end_t":end_time,
        "t_grid":grid_t,
        "x_grid":grid_x,
        }
    
    with open("msf_raw.cpkl","wb") as f:
        pickle.dump(msf_dict,f)
    
    return msf_dict

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


def cluster(det,start_time,end_time,start_x,end_x,direction, delta = 0.3, phi = 0.4):
    #%% Phase 1 tracklet prep
    ### detections have many overlaps - the first phase groups all detections that overlap sufficiently in space, and that create continuous clusters in space (i.e. iou-based tracker)
    t1 = time.time()
    ids = torch.torch.tensor([i for i in range(det.shape[0])],device = dev) # to start every detection is in a distinct cluster
    adj_list = [[] for i in range(det.shape[0])]
    
    for ts in np.arange(start_time,end_time,step = 0.1):
        elapsed = time.time() - t1
        remaining = elapsed/(ts-start_time) * (end_time - ts)
        print("\r Processing time {:.1f}/{:.1f}s     {:.2f}% done, ({:.1f}s elapsed, {:.1f}s remaining.)   ".format(ts,end_time,ts*100/(end_time-start_time),elapsed,remaining),flush = True, end = "\r")
        
        ## grab the set of detections in ts,ts+delta
        ts_idxs = torch.where(torch.logical_and(det[:,0] > ts,det[:,0] < ts+delta),1,0).nonzero().squeeze()
        ts_det = det[ts_idxs,:]
        max_idx = torch.max(ts_idxs)
        
        if ts_det.shape[0] == 0:
            continue
        first  = torch.clone(ts_det).to(dev)
        
        ## convert from state for to state-space box rcs box form
        boxes_new = torch.zeros([first.shape[0],4],device = dev)
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
            adj_list[idx] += cluster_idxs.tolist()
            
            # if cluster_idxs.shape[0] > 1: # if 1, the only match for the detection is itself
            #     # get minimum ts_idx
            #     cluster_ids = ids[cluster_idxs]
            #     min_id = torch.min(cluster_ids)
    
            #     # # find and replace each other ts_idx
            #     # for id in cluster_ids:
            #     #     ids[ids == id] = min_id
                
            #     cil = cluster_ids.shape[0]
            #     ids_exp = ids[:max_idx].unsqueeze(1).expand(max_idx,cil) # no need to consider detections we haven't seen / clustered yet
            #     cluster_ids = cluster_ids.unsqueeze(0).expand(max_idx,cil)
            #     hits = torch.where(cluster_ids == ids_exp,1,0).sum(dim = 1).nonzero().squeeze(1)
                
            #     # ids[hits] = min_id
                
    ##, now given an adjacency list for each detection, get clusters
    
    def visit(i,id = -1):
        # if this node has an id, return
        # else, assign id, and return list of neighbors
        if ids[i] != -1:
            return []
        else:
            ids[i] = id
            return adj_list[i]
        
       
        
    ids = torch.torch.tensor([-1 for i in range(det.shape[0])]) 
    next_id = 0
    
    
    t2 = time.time()
    print("\n")
    
    for i in range(ids.shape[0]):
        if i % 100 == 0: 
            elapsed = time.time() - t2
            remaining = elapsed/((i+1)/ids.shape[0])* (1- i/ids.shape[0])
            print("\rOn detection {} of {}, {:.2f}% done, ({:.1f}s elapsed, {:.1f}s remaining.) ".format(i,ids.shape[0],i/ids.shape[0]*100,elapsed,remaining),end = "\r",flush = True)
            
        if ids[i] == -1: # no assigned cluster
               ids[i] = next_id
               visited = torch.zeros(det.shape[0])
               visited[i] = 1
               to_visit = list(set(adj_list[i]))
               while len(to_visit) > 0:
                   j = to_visit.pop(0)
                   new = visit(j,id = next_id)
                   visited[j] = 1
                   for item in new:
                       if visited[item] == 0:
                           to_visit.append(item)
               next_id += 1
               
               
               
               
    
    det = det.cpu()
    ## resulting will be a set of tracklet clusters (clusters of detections)
    count = ids.unique().shape[0]
    orig_idxs =  torch.torch.tensor([i for i in range(det.shape[0])])
    out = torch.cat((det,orig_idxs.unsqueeze(1),ids.unsqueeze(1)),dim = 1)
    out = out.data.numpy()
    np.save("data/clustered_{}_{}_{}_{}.npy".format(start_time,end_time,start_x,end_x),out)
    print("\nFinished clustering detections for phase 1. Before:{}, After: {}. {:.1f}s elapsed.".format(det.shape[0],count,time.time() - t1))

def prep_tracklets(det,SHOW = True):
    
    # From detections to tracklet list
    time_idxs = torch.where(torch.logical_and(det[:,0] > start_time,det[:,0] < end_time),1,0)
    space_idxs = torch.where(torch.logical_and(det[:,1] > start_x,det[:,1]<end_x),1,0)
    direction_idxs = torch.where(torch.sign(det[:,2]) == direction,1,0)
    keep_idxs = (time_idxs * space_idxs * direction_idxs).nonzero().squeeze()
    
    det = det[keep_idxs,:]
    
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
        if len(tracklets[key]) > 1:
            tracklets[key] = torch.from_numpy(np.stack(tracklets[key]))
        
    # re-key dictionary to use consecutive 0-start integers so we can use as indices
    tracklets =  list(tracklets.values())
    
    keep = []
    for item in tracklets:
        if len(item) > 2:
            keep.append(item)        
    tracklets = keep
    print("Raveled tracklets, {} total".format(len(tracklets)))

    if SHOW:
        plt.figure(figsize = (20,15))
        colors = np.random.rand(len(tracklets),3)
        for tidx,t in enumerate(tracklets):
              if t[0,2] < -12*lane and t[0,2] > -12*(lane+1):
                  plt.plot(t[:,0],t[:,1],c = colors[tidx],linewidth=4)
        plt.title("Pre-clustering Lane {}".format(lane))
        plt.savefig("im/{}.png".format(str(0).zfill(3)))
    
        plt.show()
        
        
    return tracklets

def compute_intersections(t_threshold,x_threshold,y_threshold, index = None):
    
    if index is None:
        # get t overlaps for all tracklets
        start = time.time()
        max_t = torch.tensor([torch.max(t[:,0]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*t_threshold
        min_t = torch.tensor([torch.min(t[:,0]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*t_threshold
           
        mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
        maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
        zeros = torch.zeros(mint_int.shape,dtype=float)
        t_intersection = torch.max(zeros, maxt_int-mint_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
        
        # ensure t1 starts before t2
        t_order = torch.where(max_t.transpose(1,0) - max_t <+ 0, 1,0)
        
        
        # get x overlaps for all tracklets
        max_x = torch.tensor([torch.max(t[:,1]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*x_threshold
        min_x = torch.tensor([torch.min(t[:,1]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*x_threshold
           
        minx_int = torch.max(torch.stack((min_x,min_x.transpose(1,0)),dim = -1),dim = -1)[0]
        maxx_int = torch.min(torch.stack((max_x,max_x.transpose(1,0)),dim = -1),dim = -1)[0]
        zeros = torch.zeros(minx_int.shape,dtype=float)
        x_intersection = torch.max(zeros, maxx_int-minx_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
        
        # get y overlaps for all tracklets
        max_y = torch.tensor([torch.max(t[:,2]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*y_threshold
        min_y = torch.tensor([torch.min(t[:,2]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*y_threshold
           
        miny_int = torch.max(torch.stack((min_y,min_y.transpose(1,0)),dim = -1),dim = -1)[0]
        maxy_int = torch.min(torch.stack((max_y,max_y.transpose(1,0)),dim = -1),dim = -1)[0]
        zeros = torch.zeros(miny_int.shape,dtype=float)
        y_intersection = torch.max(zeros, maxy_int-miny_int)  # if 0, these two tracklets are not within y_threshold of one another (even disregarding time matching)
        
        
        #direction
        # direction = torch.tensor([torch.sign(t[0,2]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))
        # d_intersection = torch.where(direction * direction.transpose(1,0) == 1, 1,0)
        
        intersection = t_order * torch.where(t_intersection > 0,1,0) * torch.where(t_intersection < (t_threshold),1,0) * torch.where(x_intersection > 0,1,0)   * torch.where(y_intersection > 0,1,0) # * d_intersection
        #* torch.where(x_intersection < x_threshold,1,0)
        # zero center diagonal
        intersection = intersection * (1- torch.eye(intersection.shape[0]))
        
        print("Intersection computation took {:.2f}s".format(time.time() - start))
        yield intersection
        

        
    
   
    
    
    
   

#%% open gps file, sort and slice
if True:
    
    dev = torch.device("cuda:0")
    dev = torch.device("cpu")

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
    det = det.to(dev)
    
    print("Loading data took {:.1f}s".format(time.time() - t1))
    print("For time window [{:.1f}s,{:.1f}s], space window [{:.1f}ft,{:.1f}ft], --> {} detections".format(start_time,end_time,start_x,end_x,det.shape[0]))
    
    
    det = cluster(det,start_time,end_time,start_x,end_x,direction,delta = delta,phi = phi)
    
    


#%% Phase 2 tracklet to tracklet clustering
if True:
    start_time_total = start_time
    end_time_total = end_time
    start_x_total = start_x
    end_x_total = end_x
    det_total = np.load("data/clustered_{}_{}_{}_{}.npy".format(start_time,end_time,start_x,end_x))
    det_total = torch.from_numpy(det_total)

    
  
        

    for start_time in np.arange(start_time_total,end_time_total,step = time_chunk):
        end_time = start_time + time_chunk
        for start_x in np.arange(start_x_total,end_x_total,step = space_chunk):
            end_x = start_x + space_chunk
            
            save_file = "data/blocks/{}_{}_{}_{}_clustered.cpkl".format(start_time,end_time,start_x,end_x)
            if os.path.exists(save_file): continue
    
            print("\nWorking on sub-chunk T:{}, X:{}\n".format(start_time,start_x))
            det = det_total.clone()
            # one iteration for one chunk here
            time_idxs = torch.where(torch.logical_and(det[:,0] > start_time,det[:,0] < end_time),1,0)
            space_idxs = torch.where(torch.logical_and(det[:,1] > start_x,det[:,1]<end_x),1,0)
            direction_idxs = torch.where(torch.sign(det[:,2]) == direction,1,0)
            keep_idxs = (time_idxs * space_idxs * direction_idxs).nonzero().squeeze()
            
            det = det[keep_idxs,:]
            
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
                if len(tracklets[key]) > 1:
                    tracklets[key] = torch.from_numpy(np.stack(tracklets[key]))
                
            # re-key dictionary to use consecutive 0-start integers so we can use as indices
            tracklets =  list(tracklets.values())
            
            keep = []
            for item in tracklets:
                if len(item) > 2:
                    keep.append(item)        
            tracklets = keep
            print("Raveled tracklets, {} total".format(len(tracklets)))
            
            if False:
                print("OVERWRITING!!")
                tracklets = []
                for file in os.listdir("data/blocks"):
                    with open(os.path.join("data/blocks",file),"rb") as f:
                        d = pickle.load(f)
                        tracklets += d
                print("Loaded cached tracklet blocks, {} total".format(len(tracklets)))
            ## Compute time overlaps for each pair of tracklets
            # Each tracklet is an np array of size n_detections, 9 - timestamp, x,y,l,w,h,conf,class,original_idx,id
            #tracklets = tracklets[:2000]
            
            # TODO - vectorize
            #tgap[i,j] -  how long after tracklet i ends does tracklet j start
            # start_ts = torch.tensor([t[0,0]  for t in tracklets] ).unsqueeze(0).expand(len(tracklets),len(tracklets))
            # end_ts   = torch.tensor([t[-1,0] for t in tracklets] ).unsqueeze(1).expand(len(tracklets),len(tracklets))
            # tgap = start_ts - end_ts
            # print("Computed time gaps")
            
            # while tracklets exist that do not have source and sink  ---> This constraint ensures iteration until all tracklets are OD valid
            
            ## container for holding tracklets that have a valid source and sink
            # valid sources include: present at start, on-ramp, start of space range
            # valid sinks include: present at end, off-ramp, end of space range
            # every time we combine tracklets we check to see whether it is complete
            tracklets_complete = []
        
            if True:
                plt.figure(figsize = (20,15))
                colors = np.random.rand(len(tracklets),3)
                for tidx,t in enumerate(tracklets):
                      if t[0,2] < -12*lane and t[0,2] > -12*(lane+1):
                          plt.plot(t[:,0],t[:,1],c = colors[tidx],linewidth=4)
                plt.title("Pre-clustering Lane {}".format(lane))
                plt.savefig("im/{}.png".format(str(0).zfill(3)))
            
                plt.show()
            colors = np.random.rand(10000,3)
            colors[:,2] = 0
            
            # msf_dict = generate_msf(tracklets,
            #              start_time,end_time,
            #              start_x,end_x,
            #              grid_t=2,grid_x = 50,kwidth = 15)    
            
            for iteration in range(len(x_thresholds)-1):
        
                    
                # assign variables
                
                x_threshold           = x_thresholds[iteration]                  # tracklets beyond this distance apart from one another are not considered
                y_threshold           = y_threshold = y_thresholds[iteration]    # tracklets beyond this distance apart from one another are not considered
                t_threshold           = t_thresholds[iteration]                  # tracklets beyond this duration apart from one another are not considered
                reg_keep              = reg_keeps[iteration]                     # the last _ points are used to fit a linear regression and the first _ points from the next tracklet are scored
                min_regression_length = min_regression_lengths[iteration]        # tracklets less than _ in length are not used to fit a linear regression
                cutoff_dist           = cutoff_dists[iteration]    
                
            
                # get t overlaps for all tracklets
                start = time.time()
                max_t = torch.tensor([torch.max(t[:,0]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*t_threshold
                min_t = torch.tensor([torch.min(t[:,0]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*t_threshold
                   
                mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
                maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
                zeros = torch.zeros(mint_int.shape,dtype=float)
                t_intersection = torch.max(zeros, maxt_int-mint_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
                
                # ensure t1 starts before t2
                t_order = torch.where(max_t.transpose(1,0) - max_t <+ 0, 1,0)
                
                
                # get x overlaps for all tracklets
                max_x = torch.tensor([torch.max(t[:,1]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*x_threshold
                min_x = torch.tensor([torch.min(t[:,1]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*x_threshold
                   
                minx_int = torch.max(torch.stack((min_x,min_x.transpose(1,0)),dim = -1),dim = -1)[0]
                maxx_int = torch.min(torch.stack((max_x,max_x.transpose(1,0)),dim = -1),dim = -1)[0]
                zeros = torch.zeros(minx_int.shape,dtype=float)
                x_intersection = torch.max(zeros, maxx_int-minx_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
                
                # get y overlaps for all tracklets
                max_y = torch.tensor([torch.max(t[:,2]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*y_threshold
                min_y = torch.tensor([torch.min(t[:,2]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*y_threshold
                   
                miny_int = torch.max(torch.stack((min_y,min_y.transpose(1,0)),dim = -1),dim = -1)[0]
                maxy_int = torch.min(torch.stack((max_y,max_y.transpose(1,0)),dim = -1),dim = -1)[0]
                zeros = torch.zeros(miny_int.shape,dtype=float)
                y_intersection = torch.max(zeros, maxy_int-miny_int)  # if 0, these two tracklets are not within y_threshold of one another (even disregarding time matching)
                
                
                #direction
                # direction = torch.tensor([torch.sign(t[0,2]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))
                # d_intersection = torch.where(direction * direction.transpose(1,0) == 1, 1,0)
                
                intersection = t_order * torch.where(t_intersection > 0,1,0) * torch.where(t_intersection < (t_threshold),1,0) * torch.where(x_intersection > 0,1,0)   * torch.where(y_intersection > 0,1,0) # * d_intersection
                #* torch.where(x_intersection < x_threshold,1,0)
                # zero center diagonal
                intersection = intersection * (1- torch.eye(intersection.shape[0]))
                
               
                
                print("Intersection computation took {:.2f}s".format(time.time() - start))
                
                if False: # plot a bunch of stuff
                    for i in range(100):
                        ind = intersection[i].nonzero().squeeze()
                        
                        
                        alltr = [tracklets[i]]
                        try:
                            for j in ind:
                                #plt.scatter(tracklets[i][:,1],tracklets[i][:,2],c = tracklets[i][:,0])
                                alltr.append(tracklets[j])
                        except TypeError:
                            continue
                
                        alltr = torch.cat((alltr),dim = 0)
                        plt.scatter(alltr[:,0],alltr[:,1],c = alltr[:,2]//12)
                        plt.scatter(tracklets[i][:,0],tracklets[i][:,1],c = tracklets[i][:,2])
                        #plt.ylim([-60,0])
                        plt.title("{} total tracklets".format(len(ind)+1))
                        plt.show()
                        plt.savefig("{}.png".format(i))
                
                
                
                # Matrices for holding misalignment scores
                # align_x[i,j] the end of tracklet i pointing to the beginning of tracklet j
                # align_xb[i,j] the beginning of tracklet j pointing to the end of tracklet i 
                # thus the total score for a pair is align_x[i,j] + align_xb[i,j]
                
                
                align_x = torch.zeros(intersection.shape) + big_number
                align_y = torch.zeros(intersection.shape) + big_number
                align_xb = torch.zeros(intersection.shape) + big_number
                align_yb = torch.zeros(intersection.shape) + big_number
                
                
                ## Generate linear regression alignment scores for each pair
                SHOW = False
                for i in range(len(tracklets)):
                    
                    if tracklets[i].shape[0] < min_regression_length: continue
                   
                    # fit linear regressor
                    t1  = tracklets[i][-reg_keep:,0].unsqueeze(1)
                    y1  = tracklets[i][-reg_keep:,1:3]
                    reg = LinearRegression().fit(t1,y1)
            
                    
                    # fit linear regressor
                    t1b  = tracklets[i][:reg_keep,0].unsqueeze(1)
                    y1b  = tracklets[i][:reg_keep,1:3]
                    regb = LinearRegression().fit(t1b,y1b)
            
                    if SHOW: 
                        plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
                        #plot regression line
                        t_trend = np.array([[tracklets[i][-reg_keep,0]],[tracklets[i][-1,0]+t_threshold]])
                        y_trend = reg.predict(t_trend)
                        plt.plot(t_trend,y_trend[:,0],":",c = "k")
                        
                        #plot backward regression line
                        t_trend = np.array([[tracklets[i][reg_keep,0]],[tracklets[i][0,0]-t_threshold]])
                        y_trend = regb.predict(t_trend)
                        plt.plot(t_trend,y_trend[:,0],"--",c = "k")
                    
                    #print("On tracklet {} of {}".format(i,len(tracklets)))
                    for j in range(len(tracklets)):
                        if intersection[i,j] == 1:
                            
                            # get first bit of data
                            t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
                            y2  = tracklets[j][:reg_keep,1:3]
                            
                            pred = reg.predict(t2)
                            diff = np.abs(pred - y2.data.numpy())
                            
                            mdx = diff[:,0].mean()
                            mdy = diff[:,1].mean()
                            align_x[i,j] = mdx
                            align_y[i,j] = mdy
                            
                            if mdx > x_threshold: align_x[i,j] = big_number
                            if mdy > y_threshold: align_y[i,j] = big_number
                            
                            if SHOW: 
                                plt.scatter(tracklets[j][:,0],tracklets[j][:,1], c = "r")
            
                        if intersection[j,i] == 1:
                            t2b = tracklets[j][-reg_keep:,0].unsqueeze(1)
                            y2b = tracklets[j][-reg_keep:,1:3]
                            
                            pred = regb.predict(t2b)
                            diff = np.abs(pred - y2b.data.numpy())
                            
                            mdx = diff[:,0].mean()
                            mdy = diff[:,1].mean()
                            align_xb[j,i] = mdx
                            align_yb[j,i] = mdy
                            
                            if mdx > x_threshold: align_xb[j,i] = big_number
                            if mdy > y_threshold: align_yb[j,i] = big_number
                        
                            if SHOW: 
                                plt.scatter(tracklets[j][:,0],tracklets[j][:,1], color = (0.8,0.8,0))
            
                        # # check for short segments which will not have accurate regression lines
                        # if tracklets[i].shape[0] < min_regression_length and tracklets[j].shape[0] > min_regression_length:
                        #     align_x[i,j] = align_xb[i,j]
                        # elif tracklets[i].shape[0] > min_regression_length and tracklets[j].shape[0] < min_regression_length:
                        #     align_xb[i,j] = align_x[i,j]
                        # elif tracklets[i].shape[0] < min_regression_length and tracklets[j].shape[0] < min_regression_length:
                        #     align_x[i,j]  = big_number
                        #     align_xb[i,j] = big_number
                            
                    # find min alignment error
                    if SHOW:
                        min_idx = torch.argmin(align_x[i]**2 + align_y[i]**2)
                        min_dist = torch.sqrt(align_x[i,min_idx]**2 + align_y[i,min_idx]**2)
                        if min_dist < cutoff_dist:
                            plt.scatter(tracklets[min_idx][:,0],tracklets[min_idx][:,1], c = "b")
                            plt.title("Minimum mean distance: {:.1f}ft".format(min_dist))
                        
                        # min backwards match
                        min_idx = torch.argmin(align_xb[i]**2 + align_yb[i]**2)
                        min_dist2 = torch.sqrt(align_xb[i,min_idx]**2 + align_yb[i,min_idx]**2)
                        if min_dist2 < cutoff_dist:
                            plt.scatter(tracklets[min_idx][:,0],tracklets[min_idx][:,1], color = (0,0.7,0.7))
                            plt.title("Minimum mean distance: {:.1f}ft,{:.1f}ft".format(min_dist,min_dist2))
                        plt.show()
                        
                        
                        
                        
                        
                        
                ### Now there are initial scores for every tracklet pair
                # we next repeatedly select the best pair of tracklets and merge them, recomputing scores         
                        
        
                
             
                FINAL = False
                while len(tracklets) > 0 :
                       if len(tracklets) % 10 == 0:
                           durations = [t[-1,0] - t[0,0] for t in tracklets]
                           mean_duration = sum(durations) / len(durations)
                           print("{} Finished, {} Unfinished tracklets with mean duration: {:.1f}s".format(len(tracklets_complete),len(tracklets),mean_duration))
                    
                       ## A disgusting masking operation that should make you queasy
                       # update alignment matrices according to tracklet length logic
                       # if tracklet i is too short, but tracklet j is not, replace tracklet i score with tracklet j backwards score   align_x[i,j] <- align_xb[i,j]
                       # if tracklet j is too short but tracklet i is not, replace tracklet j backwards score with tracklet i score    align_xb[i,j] <- align_x[i,j]       
                       tracklet_lengths = torch.tensor([t.shape[0] for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))
                       mask_ilong = torch.where(tracklet_lengths > min_regression_length,1,0) # 1 where element i is long, 0 otherwise
                       mask_jlong = torch.where(tracklet_lengths.transpose(1,0) > min_regression_length,1,0) # 1 where element j is long, 0 otherwise
                
                       align_x = align_x*mask_ilong + (1-mask_ilong)*(mask_jlong*align_xb + (1-mask_jlong)*big_number)
                       align_y = align_y*mask_ilong + (1-mask_ilong)*(mask_jlong*align_yb + (1-mask_jlong)*big_number)
                
                       align_xb = align_xb*mask_jlong + (1-mask_jlong)*(mask_ilong*align_x + (1-mask_ilong)*big_number)
                       align_yb = align_yb*mask_jlong + (1-mask_jlong)*(mask_ilong*align_y + (1-mask_ilong)*big_number)
                
                
                
                
                
                
                       # compute aggregate scores as a combo of above elements
                       scores = (torch.sqrt(align_x**2 + align_y**2)  + torch.sqrt(align_xb**2 + align_yb**2)) /2
                    
                       ### select the best-scoring association - flatten these two tracklets into a single tracklet
                       # i = merged idx
                       # j = deleted idx
                       best_idx = torch.argmin(scores)
                       best_score = torch.min(scores)
                       keep = [_ for _ in range(len(tracklets))]
                       I_COMPLETE = False
                       
                       if best_score > cutoff_dist: 
                           print("WARNING: best score {:.1f} > cutoff distance {}. Consider Terminating".format(best_score,cutoff_dist))
                           
                           if FINAL:
                               break
                           else:
                               FINAL = True
                               
                               if iteration > 12:
                                   # remove all tracklets that are more or less entirely subcontained within another tracklet
                                   durations = (max_t.transpose(1,0) - min_t - t_threshold)[:,0]
                                   
                                   keep = torch.where(durations > min_duration,1,0).nonzero().squeeze(1)
                                   
                                   tracklets = [tracklets[k] for k in keep]
                               
                       if not FINAL:
                           i,j = [best_idx // len(tracklets) , best_idx % len(tracklets)]
                           if i >j: # ensure j > i for deletion purposes
                               j1 = j
                               j = i
                               i = j1
                               
                           SHOW = False
                           if SHOW: 
                               color2 = "b"
                               if FINAL: color2 = "r"
                        
                               plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
                               plt.scatter(tracklets[j][:,0],tracklets[j][:,1], c = color2)
                               plt.title(          "X:{:.1f}ft, Y:{:.1f}ft".format((align_x[i,j]+align_xb[i,j])/2   ,(align_y[i,j]+align_yb[i,j])/2    ))
                               
                               # plot the forward and backward regression lines
                               t1  = tracklets[i][-reg_keep:,0].unsqueeze(1)
                               y1  = tracklets[i][-reg_keep:,1:3]
                               reg = LinearRegression().fit(t1,y1)
                               
                               t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
                               y2  = tracklets[j][:reg_keep,1:3]
                               regb = LinearRegression().fit(t2,y2)
                               
                               t_trend = np.array([[tracklets[i][-1,0]],[tracklets[j][0,0]]])
                               y_trend = reg.predict(t_trend)
                               plt.plot(t_trend,y_trend[:,0],":",c = "k")
                               
                               y_trend2 = regb.predict(t_trend)
                               plt.plot(t_trend,y_trend2[:,0],":",color = (0.5,0.5,0.5))
                
                               
                               plt.show()
                           
                            
                           # combine two tracklet arrays
                           tracklets[i] = torch.cat((tracklets[i],tracklets[j]),dim = 0)
                           
                           # sort by time
                           tracklets[i]  = tracklets[i][tracklets[i][:,0].sort()[1]] 
                           
                           # remove larger idx from tracklets
                           del tracklets[j]
                          
                           
                
                           
                           
                           
                     
                           ## check whether the tracklet has a valid source and sink - if so remove it from further calculation
                           
                           if torch.min(tracklets[i][:,1]) < start_x + x_margin or torch.max(tracklets[i][:,0]) > end_time - t_margin:
                               if torch.max(tracklets[i][:,1]) > end_x  - x_margin or  torch.min(tracklets[i][:,0]) < start_time + t_margin:
                                   tracklets_complete.append(tracklets[i])
                                   del tracklets[i]
                                   I_COMPLETE = True
                                   print("Tracklet {} added to finished queue".format(i))
                                   
                           # update all score matrices t_intersection, x_intersection, y_intersection, intersection, x_align, y_align
                           # if tracklet i was removed, all we need to do is remove row and column i and j
                           # otherwise, we need to remove column and row j, and update column and row i
                           
                           keep.remove(j)
                           if I_COMPLETE:
                               keep.remove(i)
                       
                       # remove old matrix entries
                       max_x = max_x[keep,:][:,keep]
                       min_x = min_x[keep,:][:,keep]
                       maxx_int = maxx_int[keep,:][:,keep]
                       minx_int = minx_int[keep,:][:,keep]
                       x_intersection = x_intersection[keep,:][:,keep]
                       
                       max_y = max_y[keep,:][:,keep]
                       min_y = min_y[keep,:][:,keep]
                       maxy_int = maxy_int[keep,:][:,keep]
                       miny_int = miny_int[keep,:][:,keep]
                       y_intersection = y_intersection[keep,:][:,keep]
                       
                       max_t = max_t[keep,:][:,keep]
                       min_t = min_t[keep,:][:,keep]
                       maxt_int = maxt_int[keep,:][:,keep]
                       mint_int = mint_int[keep,:][:,keep]
                       t_intersection = t_intersection[keep,:][:,keep]
                       
                       intersection   = intersection[keep,:][:,keep]
                       align_x = align_x[keep,:][:,keep]
                       align_y = align_y[keep,:][:,keep]
                       
                       align_xb = align_xb[keep,:][:,keep]
                       align_yb = align_yb[keep,:][:,keep]
                       
                       
                       if not I_COMPLETE and not FINAL: # in this case we need to update each row and column i to reflect new detections added to it
                           max_t[:,i] = torch.max(tracklets[i][:,0]) + 0.5* t_threshold
                           min_t[i,:] = torch.min(tracklets[i][:,0]) - 0.5* t_threshold
                           mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
                           maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
                           zeros = torch.zeros(mint_int.shape,dtype=float)
                           t_intersection = torch.max(zeros, maxt_int-mint_int)
                           t_order = torch.where(max_t.transpose(1,0) - max_t < 0, 1,0)
            
            
                           max_x[:,i] = torch.max(tracklets[i][:,1]) + 0.5* x_threshold
                           min_x[i,:] = torch.min(tracklets[i][:,1]) - 0.5* x_threshold
                           minx_int = torch.max(torch.stack((min_x,min_x.transpose(1,0)),dim = -1),dim = -1)[0]
                           maxx_int = torch.min(torch.stack((max_x,max_x.transpose(1,0)),dim = -1),dim = -1)[0]
                           zeros = torch.zeros(minx_int.shape,dtype=float)
                           x_intersection = torch.max(zeros, maxx_int-minx_int)
                           
                           max_y[:,i] = torch.max(tracklets[i][:,2]) + 0.5* y_threshold
                           min_y[i,:] = torch.min(tracklets[i][:,2]) - 0.5* y_threshold
                           miny_int = torch.max(torch.stack((min_y,min_y.transpose(1,0)),dim = -1),dim = -1)[0]
                           maxy_int = torch.min(torch.stack((max_y,max_y.transpose(1,0)),dim = -1),dim = -1)[0]
                           zeros = torch.zeros(miny_int.shape,dtype=float)
                           y_intersection = torch.max(zeros, maxy_int-miny_int)
                           
                           intersection = t_order *  torch.where(t_intersection > 0,1,0) * torch.where(t_intersection < t_threshold,1,0) * torch.where(x_intersection > 0,1,0)  *  torch.where(y_intersection > 0,1,0) # * d_intersection
                           #torch.where(x_intersection < x_threshold,1,0) * 
                           # zero center diagonal
                           intersection = intersection * (1- torch.eye(intersection.shape[0]))
                           
            
                           # now we need to update align_x and align_y
                           #if tracklets[i].shape[0] < min_regression_length: continue
                           
                       # fit linear regressor
                           t1  = tracklets[i][-reg_keep:,0].unsqueeze(1)
                           y1  = tracklets[i][-reg_keep:,1:3]
                           reg = LinearRegression().fit(t1,y1)
                           
                           t1b = tracklets[i][:reg_keep,0].unsqueeze(1)
                           y1b = tracklets[i][:reg_keep,1:3]
                           regb = LinearRegression().fit(t1b,y1b)
            
                           
                           for j in range(len(tracklets)):
                               if intersection[i,j] == 1:
                                   
                                   # get first bit of data
                                   t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
                                   y2  = tracklets[j][:reg_keep,1:3]
                                   
                                   pred = reg.predict(t2)
                                   diff = np.abs(pred - y2.data.numpy())
                                   
                                   #assign
                                   mdx = diff[:,0].mean()
                                   mdy = diff[:,1].mean()
                                   align_x[i,j] = mdx
                                   align_y[i,j] = mdy
                                   
                                   #threshold
                                   if mdx > x_threshold: align_x[i,j] = 10000
                                   if mdy > y_threshold:_length: align_y[i,j] = 10000
                                   
                                   # if tracklets[i].shape[0] < min_regression_length: 
                                   #     if tracklets[j].shape[0] > min_regression_length:
                                   #         align_x[i,j] = align_xb[i,j]
                                   #         align_y[i,j] = align_yb[i,j]
                                   #     else:
                                   #         align_x[i,j] = big_number
                                   #         align_y[i,j] = big_number
                                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   # backwards regression update
                                   regb = LinearRegression().fit(t2,y2)
                                   pred = regb.predict(t1)
                                   diff = np.abs(pred - y1.data.numpy())
                                   
                                   # assign
                                   mdx = diff[:,0].mean()
                                   mdy = diff[:,1].mean()
                                   align_xb[i,j] = mdx
                                   align_yb[i,j] = mdy
                                   
                                   #threshold
                                   if mdx > x_threshold or tracklets[j].shape[0] < min_regression_length: align_xb[i,j] = 10000
                                   if mdy > y_threshold or tracklets[j].shape[0] < min_regression_length: align_yb[i,j] = 10000
                                   
                                   # if tracklets[j].shape[0] < min_regression_length: 
                                   #     if tracklets[i].shape[0] > min_regression_length:
                                   #         align_xb[i,j] = align_x[i,j]
                                   #         align_yb[i,j] = align_y[i,j]
                                   #     else:
                                   #         align_xb[i,j] = big_number
                                   #         align_yb[i,j] = big_number
                                   
                                    
                                   
                                    
                                   
                                   
                               if intersection[j,i] == 1:
                                    
                                    # get first bit of data
                                    t2  = tracklets[j][-reg_keep:,0].unsqueeze(1)
                                    y2  = tracklets[j][-reg_keep:,1:3]
                                    regj = LinearRegression().fit(t2,y2)
                                    
                                    pred = regj.predict(t1b)
                                    diff = np.abs(pred - y1b.data.numpy())
                                   
                                    # assign
                                    mdx = diff[:,0].mean()
                                    mdy = diff[:,1].mean()
                                    align_x[j,i] = mdx
                                    align_y[j,i] = mdy
                                    
                                    #threshold
                                    if mdx > x_threshold or tracklets[j].shape[0] < min_regression_length: align_x[j,i] = 10000
                                    if mdy > y_threshold or tracklets[j].shape[0] < min_regression_length: align_y[j,i] = 10000
                                    
                                    # if tracklets[j].shape[0] < min_regression_length: 
                                    #     if tracklets[i].shape[0] > min_regression_length:
                                    #         align_x[j,i] = align_xb[j,i]
                                    #         align_y[j,i] = align_yb[j,i]
                                    #     else:
                                    #         align_x[j,i] = big_number
                                    #         align_y[j,i] = big_number
                                            
                                            
                                            
                                            
                                            
                    
                                    # update backwards regression too
                                    pred = regb.predict(t2)
                                    diff = np.abs(pred - y2.data.numpy())
                                    
                                    #assign
                                    mdx = diff[:,0].mean()
                                    mdy = diff[:,1].mean()
                                    align_xb[j,i] = mdx
                                    align_yb[j,i] = mdy
                                    
                                    #threshold
                                    if mdx > x_threshold or tracklets[i].shape[0] < min_regression_length: align_xb[j,i] = 10000
                                    if mdy > y_threshold or tracklets[i].shape[0] < min_regression_length: align_yb[j,i] = 10000
                                    
                                    # if tracklets[i].shape[0] < min_regression_length: 
                                    #     if tracklets[j].shape[0] > min_regression_length:
                                    #         align_xb[j,i] = align_x[j,i]
                                    #         align_yb[j,i] = align_y[j,i]
                                    #     else:
                                    #         align_xb[j,i] = big_number
                                    #         align_yb[j,i] = big_number
                                    
                    
                               # ensure no matches beyond threshold
                               #align_x = torch.where(align_x > x_threshold,10000,align_x)
                               #align_y = torch.where(align_y > y_threshold,10000,align_y)
                            
                 # intersection score - we could use a rasterized roadway map - only one vehicle can occupy each 1 foot by 1 foot grid cell for a timestep - 1 foot grid by 10 Hz = 86 GB ...
                 # initially, every trajectory is painted into this
            
                 # alternatively - for every pair where at least one tracklet is physically and temporally proximal
                  # determine whether a linear interpolation between that pair intersects this trajectory
                  # if we do this, then when two tracklets are combined, their intersection score is an OR operation - if either tracklet A or B could not be paired with tracklet C because of an intersection, then AB cannot be paired with C\
                
                if False:
                    plt.figure(figsize = (20,15))
                    for tidx,t in enumerate(tracklets):
                        if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
                            plt.plot(t[:,0],t[:,1],c = colors[tidx],linewidth = 4)
                            plt.text(t[0,0],t[0,1],tidx)
                    for tidx,t in enumerate(tracklets_complete):
                        if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
                            plt.plot(t[:,0],t[:,1],c = "b",linewidth = 4)
                    plt.title("Post clustering lane {} after pass {}".format(lane,iteration))
                    plt.savefig("im/{}.png".format(str(iteration+1).zfill(3)))
                    plt.show()
            
                
        #%% Phase 3 - tracklet completion using mean speed field   
            
            msf_dict = generate_msf(tracklets+tracklets_complete,
                         start_time,end_time,
                         start_x,end_x,
                         grid_t=2,grid_x = 50,kwidth = 9)
            
            # for i in range(100):
            #     virtual = get_msf_points(tracklets[i],msf_dict)
            #     plot_tracklet(tracklets,i,virtual = virtual)
            
            x_threshold           = x_thresholds[-1]                  # tracklets beyond this distance apart from one another are not considered
            y_threshold           = y_threshold = y_thresholds[-1]    # tracklets beyond this distance apart from one another are not considered
            t_threshold           = t_thresholds[-1]                  # tracklets beyond this duration apart from one another are not considered
            reg_keep              = reg_keeps[-1]                     # the last _ points are used to fit a linear regression and the first _ points from the next tracklet are scored
            min_regression_length = min_regression_lengths[-1]        # tracklets less than _ in length are not used to fit a linear regression
            cutoff_dist           = cutoff_dists[-1]        
            
            # get t overlaps for all tracklets
            start = time.time()
            max_t = torch.tensor([torch.max(t[:,0]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*t_threshold
            min_t = torch.tensor([torch.min(t[:,0]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*t_threshold
               
            mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
            maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
            zeros = torch.zeros(mint_int.shape,dtype=float)
            t_intersection = torch.max(zeros, maxt_int-mint_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
            
            # ensure t1 starts before t2
            t_order = torch.where(max_t.transpose(1,0) - max_t <+ 0, 1,0)
            
            
            # get x overlaps for all tracklets
            max_x = torch.tensor([torch.max(t[:,1]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*x_threshold
            min_x = torch.tensor([torch.min(t[:,1]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*x_threshold
               
            minx_int = torch.max(torch.stack((min_x,min_x.transpose(1,0)),dim = -1),dim = -1)[0]
            maxx_int = torch.min(torch.stack((max_x,max_x.transpose(1,0)),dim = -1),dim = -1)[0]
            zeros = torch.zeros(minx_int.shape,dtype=float)
            x_intersection = torch.max(zeros, maxx_int-minx_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
            
            # get y overlaps for all tracklets
            max_y = torch.tensor([torch.max(t[:,2]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*y_threshold
            min_y = torch.tensor([torch.min(t[:,2]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*y_threshold
               
            miny_int = torch.max(torch.stack((min_y,min_y.transpose(1,0)),dim = -1),dim = -1)[0]
            maxy_int = torch.min(torch.stack((max_y,max_y.transpose(1,0)),dim = -1),dim = -1)[0]
            zeros = torch.zeros(miny_int.shape,dtype=float)
            y_intersection = torch.max(zeros, maxy_int-miny_int)  # if 0, these two tracklets are not within y_threshold of one another (even disregarding time matching)
            
            
            #direction
            # direction = torch.tensor([torch.sign(t[0,2]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))
            # d_intersection = torch.where(direction * direction.transpose(1,0) == 1, 1,0)
            
            intersection = t_order * torch.where(t_intersection > 0,1,0) * torch.where(t_intersection < (t_threshold+t_buffer),1,0) * torch.where(x_intersection > 0,1,0)   * torch.where(y_intersection > 0,1,0) # * d_intersection
            #* torch.where(x_intersection < x_threshold,1,0)
            # zero center diagonal
            intersection = intersection * (1- torch.eye(intersection.shape[0]))
            
            
            
            
            align_x = torch.zeros(intersection.shape) + big_number
            SHOW = False
            # generate initial virtual trajectory scores
            for i in range(len(tracklets)):
                if SHOW: plt.figure()
                
                # get virtual points
                virtual = get_msf_points(tracklets[i],msf_dict)
                
                if SHOW:
                    plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
                    plt.plot(virtual[:,0],virtual[:,1],":",color = (0.2,0.2,0.2))
                
               
                # compare to each candidate match
                for j in range(len(tracklets)):
                    if intersection[i,j] == 1:
                        
                       # nix this # for each point in tracklet [j] within some threshold of the times within virtual, find closest time point, compute x and y distance, and add to score
                       # lets do it incredibly crudely, find first point of tracklet j, find closest virtual trajectory point by time, compute offset
                       jx = tracklets[j][0,1]
                       jt = tracklets[j][0,0]
                       match_idx = torch.argmin(torch.abs(virtual[:,0] - jt))
                       align_x[i,j] = torch.abs(jx - virtual[match_idx,1])
                    
                       if SHOW: 
                            plt.scatter(tracklets[j][:,0],tracklets[j][:,1], color = (0.8,0.8,0))
        
                        
                # find min alignment error
                if SHOW:
                    min_idx = torch.argmin(align_x[i]**2 + align_y[i]**2)
                    min_dist = torch.sqrt(align_x[i,min_idx]**2 + align_y[i,min_idx]**2)
                    if min_dist < cutoff_dist:
                        plt.scatter(tracklets[min_idx][:,0],tracklets[min_idx][:,1], c = "b")
                        plt.title("Minimum mean distance: {:.1f}ft".format(min_dist))
                        plt.show()
                 
                
                 
                
           
                
            
            
            
            
            
            
            
            
            
           
            
            
        
            FINAL = False
            while len(tracklets) > 0 :
                   if len(tracklets) % 10 == 0:
                       durations = [t[-1,0] - t[0,0] for t in tracklets]
                       mean_duration = sum(durations) / len(durations)
                       print("{} Finished, {} Unfinished tracklets with mean duration: {:.1f}s".format(len(tracklets_complete),len(tracklets),mean_duration))
                
                   ## A disgusting masking operation that should make you queasy
                   # update alignment matrices according to tracklet length logic
                   # if tracklet i is too short, but tracklet j is not, replace tracklet i score with tracklet j backwards score   align_x[i,j] <- align_xb[i,j]
                   # if tracklet j is too short but tracklet i is not, replace tracklet j backwards score with tracklet i score    align_xb[i,j] <- align_x[i,j]       
            
            
            
            
            
            
                   # compute aggregate scores as a combo of above elements
                   scores = torch.sqrt(align_x**2 + align_y**2) 
                
                   ### select the best-scoring association - flatten these two tracklets into a single tracklet
                   # i = merged idx
                   # j = deleted idx
                   best_idx = torch.argmin(scores)
                   best_score = torch.min(scores)
                   keep = [_ for _ in range(len(tracklets))]
        
                   if best_score > cutoff_dist: 
                       print("WARNING: best score {:.1f} > cutoff distance {}. Consider Terminating".format(best_score,cutoff_dist))
                       
                       if FINAL:
                           break
                       else:
                           FINAL = True
                           
                        
                           
                   if not FINAL:
                       i,j = [best_idx // len(tracklets) , best_idx % len(tracklets)]
            
                       SHOW = False
                       if SHOW: 
                           color2 = "b"
                           if FINAL: color2 = "r"
                    
                           plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
                           plt.scatter(tracklets[j][:,0],tracklets[j][:,1], c = color2)
                           plt.title(          "X:{:.1f}ft, Y:{:.1f}ft".format(align_x[i,j],align_y[i,j]))
                           
                           plt.show()
                       
                        
                       # combine two tracklet arrays
                       tracklets[i] = torch.cat((tracklets[i],tracklets[j]),dim = 0)
                       
                       # sort by time
                       tracklets[i]  = tracklets[i][tracklets[i][:,0].sort()[1]] 
                       
                       # remove larger idx from tracklets
                       del tracklets[j]
                      
                       
            
                       
                       
                       
                 
                       ## check whether the tracklet has a valid source and sink - if so remove it from further calculation
                       I_COMPLETE = False
                       if torch.min(tracklets[i][:,1]) < start_x + x_margin or torch.max(tracklets[i][:,0]) > end_time - t_margin:
                           if torch.max(tracklets[i][:,1]) > end_x  - x_margin or  torch.min(tracklets[i][:,0]) < start_time + t_margin:
                               tracklets_complete.append(tracklets[i])
                               del tracklets[i]
                               I_COMPLETE = True
                               print("Tracklet {} added to finished queue".format(i))
                               
                       # update all score matrices t_intersection, x_intersection, y_intersection, intersection, x_align, y_align
                       # if tracklet i was removed, all we need to do is remove row and column i and j
                       # otherwise, we need to remove column and row j, and update column and row i
                       
                       keep.remove(j)
                       if I_COMPLETE:
                           keep.remove(i)
                   
                   # remove old matrix entries
                   max_x = max_x[keep,:][:,keep]
                   min_x = min_x[keep,:][:,keep]
                   maxx_int = maxx_int[keep,:][:,keep]
                   minx_int = minx_int[keep,:][:,keep]
                   x_intersection = x_intersection[keep,:][:,keep]
                   
                   max_y = max_y[keep,:][:,keep]
                   min_y = min_y[keep,:][:,keep]
                   maxy_int = maxy_int[keep,:][:,keep]
                   miny_int = miny_int[keep,:][:,keep]
                   y_intersection = y_intersection[keep,:][:,keep]
                   
                   max_t = max_t[keep,:][:,keep]
                   min_t = min_t[keep,:][:,keep]
                   maxt_int = maxt_int[keep,:][:,keep]
                   mint_int = mint_int[keep,:][:,keep]
                   t_intersection = t_intersection[keep,:][:,keep]
                   
                   intersection   = intersection[keep,:][:,keep]
                   align_x = align_x[keep,:][:,keep]
                   align_y = align_y[keep,:][:,keep]
                   
                  
                   
                   if not I_COMPLETE and not FINAL: # in this case we need to update each row and column i to reflect new detections added to it
                       max_t[:,i] = torch.max(tracklets[i][:,0]) + 0.5* t_threshold
                       min_t[i,:] = torch.min(tracklets[i][:,0]) - 0.5* t_threshold
                       mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
                       maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
                       zeros = torch.zeros(mint_int.shape,dtype=float)
                       t_intersection = torch.max(zeros, maxt_int-mint_int)
                       t_order = torch.where(max_t.transpose(1,0) - max_t < 0, 1,0)
        
        
                       max_x[:,i] = torch.max(tracklets[i][:,1]) + 0.5* x_threshold
                       min_x[i,:] = torch.min(tracklets[i][:,1]) - 0.5* x_threshold
                       minx_int = torch.max(torch.stack((min_x,min_x.transpose(1,0)),dim = -1),dim = -1)[0]
                       maxx_int = torch.min(torch.stack((max_x,max_x.transpose(1,0)),dim = -1),dim = -1)[0]
                       zeros = torch.zeros(minx_int.shape,dtype=float)
                       x_intersection = torch.max(zeros, maxx_int-minx_int)
                       
                       max_y[:,i] = torch.max(tracklets[i][:,2]) + 0.5* y_threshold
                       min_y[i,:] = torch.min(tracklets[i][:,2]) - 0.5* y_threshold
                       miny_int = torch.max(torch.stack((min_y,min_y.transpose(1,0)),dim = -1),dim = -1)[0]
                       maxy_int = torch.min(torch.stack((max_y,max_y.transpose(1,0)),dim = -1),dim = -1)[0]
                       zeros = torch.zeros(miny_int.shape,dtype=float)
                       y_intersection = torch.max(zeros, maxy_int-miny_int)
                       
                       intersection = t_order *  torch.where(t_intersection > 0,1,0) * torch.where(t_intersection < (t_threshold+t_buffer),1,0) * torch.where(x_intersection > 0,1,0)  *  torch.where(y_intersection > 0,1,0) # * d_intersection
                       #torch.where(x_intersection < x_threshold,1,0) * 
                       # zero center diagonal
                       intersection = intersection * (1- torch.eye(intersection.shape[0]))
                       
        
                       # now we need to update align_x and align_y
        
        
                       virtual =  get_msf_points(tracklets[i],msf_dict)
                       for j in range(len(tracklets)):
                           if intersection[i,j] == 1:
                               
                              
                               # lets do it incredibly crudely, find first point of tracklet j, find closest virtual trajectory point by time, compute offset
                               jx = tracklets[j][0,1]
                               jt = tracklets[j][0,0]
                               match_idx = torch.argmin(torch.abs(virtual[:,0] - jt))
                               mdx = torch.abs(jx - virtual[match_idx,1])
                               
                               # get first bit of data
                               t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
                               y2  = tracklets[j][:reg_keep,1:3]
                               
                               pred = reg.predict(t2)
                               diff = np.abs(pred - y2.data.numpy())
                               
                               #assign
                               mdy = diff[:,1].mean()
                               align_y[i,j] = mdy
                               align_x[i,j] = mdx
                              
                               #threshold
                               if mdx > x_threshold: align_x[i,j] = 10000
                               if mdy > y_threshold:_length: align_y[i,j] = 10000
                               
                               
                           # if intersection[j,i] == 1:
                                
                           #      # get first bit of data
                           #      t2  = tracklets[j][-reg_keep:,0].unsqueeze(1)
                           #      y2  = tracklets[j][-reg_keep:,1:3]
                           #      regj = LinearRegression().fit(t2,y2)
                                
                           #      pred = regj.predict(t1b)
                           #      diff = np.abs(pred - y1b.data.numpy())
                               
                           #      # assign
                           #      mdx = diff[:,0].mean()
                           #      mdy = diff[:,1].mean()
                           #      align_x[j,i] = mdx
                           #      align_y[j,i] = mdy
                                
                           #      #threshold
                           #      if mdx > x_threshold or tracklets[j].shape[0] < min_regression_length: align_x[j,i] = 10000
                           #      if mdy > y_threshold or tracklets[j].shape[0] < min_regression_length: align_y[j,i] = 10000
                                
                           #      # if tracklets[j].shape[0] < min_regression_length: 
                           #      #     if tracklets[i].shape[0] > min_regression_length:
                           #      #         align_x[j,i] = align_xb[j,i]
                           #      #         align_y[j,i] = align_yb[j,i]
                           #      #     else:
                           #      #         align_x[j,i] = big_number
                           #      #         align_y[j,i] = big_number
                                        
                                        
                                        
                                        
                                        
                
                           #      # update backwards regression too
                           #      pred = regb.predict(t2)
                           #      diff = np.abs(pred - y2.data.numpy())
                                
                           #      #assign
                           #      mdx = diff[:,0].mean()
                           #      mdy = diff[:,1].mean()
                           #      align_xb[j,i] = mdx
                           #      align_yb[j,i] = mdy
                                
                           #      #threshold
                           #      if mdx > x_threshold or tracklets[i].shape[0] < min_regression_length: align_xb[j,i] = 10000
                           #      if mdy > y_threshold or tracklets[i].shape[0] < min_regression_length: align_yb[j,i] = 10000
                                
        
            # save data
            with open(save_file,"wb") as f:
                pickle.dump(tracklets + tracklets_complete,f)
        
            plt.figure(figsize = (20,15))
            for tidx,t in enumerate(tracklets):
                if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
                    plt.plot(t[:,0],t[:,1],c = colors[tidx],linewidth = 4)
                    plt.text(t[0,0],t[0,1],tidx)
            for tidx,t in enumerate(tracklets_complete):
                if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
                    plt.plot(t[:,0],t[:,1],c = "b",linewidth = 4)
            plt.title("Post clustering lane {} after pass {}".format(lane,iteration))
            plt.savefig("im/{}.png".format(str(iteration+1).zfill(3)))
            plt.show()


            # finally, we check for intersecting tracklets
            max_t = torch.tensor([torch.max(t[:,0]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  
            min_t = torch.tensor([torch.min(t[:,0]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  
            mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
            maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
            zeros = torch.zeros(mint_int.shape,dtype=float)
            t_intersection = torch.max(zeros, maxt_int-mint_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
            
            intersection_dist = torch.zeros(t_intersection.shape) + big_number
            # go through all pairs and compute mean distance over intersection
            for i in range(len(tracklets)):
                if i >= t_intersection.shape[0]: break
                print("On tracklet {}".format(i))
                for j in range(i+1,len(tracklets)):
                    if j == i: continue
                    if j >= t_intersection.shape[0]: continue
                    
                    hz = 0.1
                    intersection_threshold = 20

                    if t_intersection[i,j] > 0:
                        tmin = mint_int[i,j]
                        tmax = maxt_int[i,j]
                        
                        
                        # compute mean dist over t_overlap range at X Hz
                        try: eval_t = torch.arange(tmin,tmax-hz,step = hz)
                        except RuntimeError:
                            eval_t = [tmin,tmax]
                        
                        i_idxs = []
                        j_idxs = []
                        
                        i_iter = 0
                        j_iter = 0
                        
                        for t in eval_t:
                            while tracklets[i][i_iter,0] < t: i_iter += 1
                            i_idxs.append(i_iter)
                            
                            while tracklets[j][j_iter,0] < t: j_iter += 1
                            j_idxs.append(j_iter)
                    
                        # only use close comparison points to compute score
                        mask = torch.where(tracklets[i][i_idxs,0]- tracklets[j][j_idxs,0] < 0.1,1,0).nonzero().squeeze(1)
                        ix = tracklets[i][i_idxs,1][mask]
                        iy = tracklets[i][i_idxs,2][mask]
                        jx = tracklets[j][j_idxs,1][mask]
                        jy = tracklets[j][j_idxs,2][mask]
                        mean_dist = torch.sqrt((ix-jx)**2 + (iy-jy)**2).mean()
                        
                        intersection_dist[i,j] = mean_dist
                        if mean_dist < intersection_threshold:
                            print("Combining tracklets {} and {} with mean intersecting distance {:.1f}ft and {} comparisons".format(i,j,mean_dist,len(i_idxs)))
                        
            #                 # same raveling process as above
            #                 # combine two tracklet arrays
            #                 tracklets[i] = torch.cat((tracklets[i],tracklets[j]),dim = 0)
                            
            #                 # sort by time
            #                 tracklets[i]  = tracklets[i][tracklets[i][:,0].sort()[1]] 
                            
            #                 # remove larger idx from tracklets
            #                 del tracklets[j]
            #                 print(len(tracklets))
                            
                 
                            
                            
                            
                      
            #                 ## check whether the tracklet has a valid source and sink - if so remove it from further calculation
            #                 I_COMPLETE = False
            #                 if torch.min(tracklets[i][:,1]) < start_x + x_margin or torch.max(tracklets[i][:,0]) > end_time - t_margin:
            #                     if torch.max(tracklets[i][:,1]) > end_x  - x_margin or  torch.min(tracklets[i][:,0]) < start_time + t_margin:
            #                         tracklets_complete.append(tracklets[i])
            #                         del tracklets[i]
            #                         I_COMPLETE = True
            #                         print("Tracklet {} added to finished queue".format(i))
                                    
            #                 # update all score matrices t_intersection, x_intersection, y_intersection, intersection, x_align, y_align
            #                 # if tracklet i was removed, all we need to do is remove row and column i and j
            #                 # otherwise, we need to remove column and row j, and update column and row i
            #                 keep = [_ for _ in range(len(tracklets))]
                            
            #                 keep.remove(j)
            #                 if I_COMPLETE:
            #                     keep.remove(i)
            
                                
            #                 # remove old matrix entries
            #                 max_x = max_x[keep,:][:,keep]
            #                 min_x = min_x[keep,:][:,keep]
            #                 maxx_int = maxx_int[keep,:][:,keep]
            #                 minx_int = minx_int[keep,:][:,keep]
            #                 x_intersection = x_intersection[keep,:][:,keep]
                            
            #                 max_y = max_y[keep,:][:,keep]
            #                 min_y = min_y[keep,:][:,keep]
            #                 maxy_int = maxy_int[keep,:][:,keep]
            #                 miny_int = miny_int[keep,:][:,keep]
            #                 y_intersection = y_intersection[keep,:][:,keep]
                            
            #                 max_t = max_t[keep,:][:,keep]
            #                 min_t = min_t[keep,:][:,keep]
            #                 maxt_int = maxt_int[keep,:][:,keep]
            #                 mint_int = mint_int[keep,:][:,keep]
            #                 t_intersection = t_intersection[keep,:][:,keep]
                            
            #                 intersection   = intersection[keep,:][:,keep]
            #                 align_x = align_x[keep,:][:,keep]
            #                 align_y = align_y[keep,:][:,keep]
                            
                           
                            
            #                 if not I_COMPLETE and not FINAL: # in this case we need to update each row and column i to reflect new detections added to it
            #                     max_t[:,i] = torch.max(tracklets[i][:,0]) + 0.5* t_threshold
            #                     min_t[i,:] = torch.min(tracklets[i][:,0]) - 0.5* t_threshold
            #                     mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
            #                     maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
            #                     zeros = torch.zeros(mint_int.shape,dtype=float)
            #                     t_intersection = torch.max(zeros, maxt_int-mint_int)
            #                     t_order = torch.where(max_t.transpose(1,0) - max_t < 0, 1,0)
                            
                            
        
        
        
            # # save data
            # with open(save_file,"wb") as f:
            #     pickle.dump(tracklets + tracklets_complete,f)
        
            # plt.figure(figsize = (20,15))
            # for tidx,t in enumerate(tracklets):
            #     if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
            #         plt.plot(t[:,0],t[:,1],c = colors[tidx],linewidth = 4)
            #         plt.text(t[0,0],t[0,1],tidx)
            # for tidx,t in enumerate(tracklets_complete):
            #     if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
            #         plt.plot(t[:,0],t[:,1],c = "b",linewidth = 4)
            # plt.title("Post clustering lane {} after pass {}".format(lane,iteration))
            # plt.savefig("im/{}.png".format(str(iteration+1).zfill(3)))
            # plt.show()



#%% Phase 2 tracklet to tracklet clustering but at larger scale from pre-saved files
if False:
    start_time_total = start_time
    end_time_total = end_time
    start_x_total = start_x
    end_x_total = end_x


    
  
        

    for start_time in np.arange(start_time_total,end_time_total,step = time_chunk):
        end_time = start_time + time_chunk
        for start_x in np.arange(start_x_total,end_x_total,step = space_chunk):
            t_buffer = 0
            end_x = start_x + space_chunk
            
            # load all relevant chunks
            save_file = "data/blocks/{}_{}_{}_{}_clustered.cpkl".format(start_time,end_time,start_x,end_x)
            if os.path.exists(save_file): continue
    
            print("\nWorking on sub-chunk T:{}-{}, X:{}-{}\n".format(start_time,end_time,start_x,end_x))

            
            tracklets = []
            for file in os.listdir("data/blocks"):
                if int(file.split("_")[0]) >= start_time and int(file.split("_")[1]) <= end_time and int(file.split("_")[2]) >= start_x and int(file.split("_")[3]) <= end_x:
                    with open(os.path.join("data/blocks",file),"rb") as f:
                        d = pickle.load(f)
                        tracklets += d
            print("Loaded cached tracklet blocks, {} total".format(len(tracklets)))
            ## Compute time overlaps for each pair of tracklets
            # Each tracklet is an np array of size n_detections, 9 - timestamp, x,y,l,w,h,conf,class,original_idx,id
            #tracklets = tracklets[:2000]
            
            # TODO - vectorize
            #tgap[i,j] -  how long after tracklet i ends does tracklet j start
            # start_ts = torch.tensor([t[0,0]  for t in tracklets] ).unsqueeze(0).expand(len(tracklets),len(tracklets))
            # end_ts   = torch.tensor([t[-1,0] for t in tracklets] ).unsqueeze(1).expand(len(tracklets),len(tracklets))
            # tgap = start_ts - end_ts
            # print("Computed time gaps")
            
            # while tracklets exist that do not have source and sink  ---> This constraint ensures iteration until all tracklets are OD valid
            
            ## container for holding tracklets that have a valid source and sink
            # valid sources include: present at start, on-ramp, start of space range
            # valid sinks include: present at end, off-ramp, end of space range
            # every time we combine tracklets we check to see whether it is complete
            tracklets_complete = []
        
            if False:
                plt.figure(figsize = (20,15))
                colors = np.random.rand(len(tracklets),3)
                for tidx,t in enumerate(tracklets):
                      if t[0,2] < -12*lane and t[0,2] > -12*(lane+1):
                          plt.plot(t[:,0],t[:,1],c = colors[tidx],linewidth=4)
                plt.title("Pre-clustering Lane {}".format(lane))
                plt.savefig("im/{}.png".format(str(0).zfill(3)))
            
                plt.show()
            colors = np.random.rand(10000,3)
            colors[:,2] = 0
            
            # msf_dict = generate_msf(tracklets,
            #              start_time,end_time,
            #              start_x,end_x,
            #              grid_t=2,grid_x = 50,kwidth = 15)    
            
            for iteration in range(len(x_thresholds)-2,len(x_thresholds)-1):
                
                    
                # assign variables
                
                x_threshold           = x_thresholds[iteration]                  # tracklets beyond this distance apart from one another are not considered
                y_threshold           = y_threshold = y_thresholds[iteration]    # tracklets beyond this distance apart from one another are not considered
                t_threshold           = t_thresholds[iteration]                  # tracklets beyond this duration apart from one another are not considered
                reg_keep              = reg_keeps[iteration]                     # the last _ points are used to fit a linear regression and the first _ points from the next tracklet are scored
                min_regression_length = min_regression_lengths[iteration]        # tracklets less than _ in length are not used to fit a linear regression
                cutoff_dist           = cutoff_dists[iteration]    
                
            
                # get t overlaps for all tracklets
                start = time.time()
                max_t = torch.tensor([torch.max(t[:,0]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*t_threshold
                min_t = torch.tensor([torch.min(t[:,0]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*t_threshold
                   
                mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
                maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
                zeros = torch.zeros(mint_int.shape,dtype=float)
                t_intersection = torch.max(zeros, maxt_int-mint_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
                
                # ensure t1 starts before t2
                t_order = torch.where(max_t.transpose(1,0) - max_t <+ 0, 1,0)
                
                
                # get x overlaps for all tracklets
                max_x = torch.tensor([torch.max(t[:,1]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*x_threshold
                min_x = torch.tensor([torch.min(t[:,1]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*x_threshold
                   
                minx_int = torch.max(torch.stack((min_x,min_x.transpose(1,0)),dim = -1),dim = -1)[0]
                maxx_int = torch.min(torch.stack((max_x,max_x.transpose(1,0)),dim = -1),dim = -1)[0]
                zeros = torch.zeros(minx_int.shape,dtype=float)
                x_intersection = torch.max(zeros, maxx_int-minx_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
                
                # get y overlaps for all tracklets
                max_y = torch.tensor([torch.max(t[:,2]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*y_threshold
                min_y = torch.tensor([torch.min(t[:,2]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*y_threshold
                   
                miny_int = torch.max(torch.stack((min_y,min_y.transpose(1,0)),dim = -1),dim = -1)[0]
                maxy_int = torch.min(torch.stack((max_y,max_y.transpose(1,0)),dim = -1),dim = -1)[0]
                zeros = torch.zeros(miny_int.shape,dtype=float)
                y_intersection = torch.max(zeros, maxy_int-miny_int)  # if 0, these two tracklets are not within y_threshold of one another (even disregarding time matching)
                
                
                #direction
                # direction = torch.tensor([torch.sign(t[0,2]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))
                # d_intersection = torch.where(direction * direction.transpose(1,0) == 1, 1,0)
                
                intersection = t_order * torch.where(t_intersection > 0,1,0) * torch.where(t_intersection < (t_threshold),1,0) * torch.where(x_intersection > 0,1,0)   * torch.where(y_intersection > 0,1,0) # * d_intersection
                #* torch.where(x_intersection < x_threshold,1,0)
                # zero center diagonal
                intersection = intersection * (1- torch.eye(intersection.shape[0]))
                
               
                
                print("Intersection computation took {:.2f}s".format(time.time() - start))
                
                if False: # plot a bunch of stuff
                    for i in range(100):
                        ind = intersection[i].nonzero().squeeze()
                        
                        
                        alltr = [tracklets[i]]
                        try:
                            for j in ind:
                                #plt.scatter(tracklets[i][:,1],tracklets[i][:,2],c = tracklets[i][:,0])
                                alltr.append(tracklets[j])
                        except TypeError:
                            continue
                
                        alltr = torch.cat((alltr),dim = 0)
                        plt.scatter(alltr[:,0],alltr[:,1],c = alltr[:,2]//12)
                        plt.scatter(tracklets[i][:,0],tracklets[i][:,1],c = tracklets[i][:,2])
                        #plt.ylim([-60,0])
                        plt.title("{} total tracklets".format(len(ind)+1))
                        plt.show()
                        plt.savefig("{}.png".format(i))
                
                
                
                # Matrices for holding misalignment scores
                # align_x[i,j] the end of tracklet i pointing to the beginning of tracklet j
                # align_xb[i,j] the beginning of tracklet j pointing to the end of tracklet i 
                # thus the total score for a pair is align_x[i,j] + align_xb[i,j]
                
                
                align_x = torch.zeros(intersection.shape) + big_number
                align_y = torch.zeros(intersection.shape) + big_number
                align_xb = torch.zeros(intersection.shape) + big_number
                align_yb = torch.zeros(intersection.shape) + big_number
                
                
                ## Generate linear regression alignment scores for each pair
                SHOW = False
                for i in range(len(tracklets)):
                    
                    if tracklets[i].shape[0] < min_regression_length: continue
                   
                    # fit linear regressor
                    t1  = tracklets[i][-reg_keep:,0].unsqueeze(1)
                    y1  = tracklets[i][-reg_keep:,1:3]
                    reg = LinearRegression().fit(t1,y1)
            
                    
                    # fit linear regressor
                    t1b  = tracklets[i][:reg_keep,0].unsqueeze(1)
                    y1b  = tracklets[i][:reg_keep,1:3]
                    regb = LinearRegression().fit(t1b,y1b)
            
                    if SHOW: 
                        plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
                        #plot regression line
                        t_trend = np.array([[tracklets[i][-reg_keep,0]],[tracklets[i][-1,0]+t_threshold]])
                        y_trend = reg.predict(t_trend)
                        plt.plot(t_trend,y_trend[:,0],":",c = "k")
                        
                        #plot backward regression line
                        t_trend = np.array([[tracklets[i][reg_keep,0]],[tracklets[i][0,0]-t_threshold]])
                        y_trend = regb.predict(t_trend)
                        plt.plot(t_trend,y_trend[:,0],"--",c = "k")
                    
                    print("On tracklet {} of {}".format(i,len(tracklets)))
                    for j in range(len(tracklets)):
                        if intersection[i,j] == 1:
                            
                            # get first bit of data
                            t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
                            y2  = tracklets[j][:reg_keep,1:3]
                            
                            pred = reg.predict(t2)
                            diff = np.abs(pred - y2.data.numpy())
                            
                            mdx = diff[:,0].mean()
                            mdy = diff[:,1].mean()
                            align_x[i,j] = mdx
                            align_y[i,j] = mdy
                            
                            if mdx > x_threshold: align_x[i,j] = big_number
                            if mdy > y_threshold: align_y[i,j] = big_number
                            
                            if SHOW: 
                                plt.scatter(tracklets[j][:,0],tracklets[j][:,1], c = "r")
            
                        if intersection[j,i] == 1:
                            t2b = tracklets[j][-reg_keep:,0].unsqueeze(1)
                            y2b = tracklets[j][-reg_keep:,1:3]
                            
                            pred = regb.predict(t2b)
                            diff = np.abs(pred - y2b.data.numpy())
                            
                            mdx = diff[:,0].mean()
                            mdy = diff[:,1].mean()
                            align_xb[j,i] = mdx
                            align_yb[j,i] = mdy
                            
                            if mdx > x_threshold: align_xb[j,i] = big_number
                            if mdy > y_threshold: align_yb[j,i] = big_number
                        
                            if SHOW: 
                                plt.scatter(tracklets[j][:,0],tracklets[j][:,1], color = (0.8,0.8,0))
            
                        # # check for short segments which will not have accurate regression lines
                        # if tracklets[i].shape[0] < min_regression_length and tracklets[j].shape[0] > min_regression_length:
                        #     align_x[i,j] = align_xb[i,j]
                        # elif tracklets[i].shape[0] > min_regression_length and tracklets[j].shape[0] < min_regression_length:
                        #     align_xb[i,j] = align_x[i,j]
                        # elif tracklets[i].shape[0] < min_regression_length and tracklets[j].shape[0] < min_regression_length:
                        #     align_x[i,j]  = big_number
                        #     align_xb[i,j] = big_number
                            
                    # find min alignment error
                    if SHOW:
                        min_idx = torch.argmin(align_x[i]**2 + align_y[i]**2)
                        min_dist = torch.sqrt(align_x[i,min_idx]**2 + align_y[i,min_idx]**2)
                        if min_dist < cutoff_dist:
                            plt.scatter(tracklets[min_idx][:,0],tracklets[min_idx][:,1], c = "b")
                            plt.title("Minimum mean distance: {:.1f}ft".format(min_dist))
                        
                        # min backwards match
                        min_idx = torch.argmin(align_xb[i]**2 + align_yb[i]**2)
                        min_dist2 = torch.sqrt(align_xb[i,min_idx]**2 + align_yb[i,min_idx]**2)
                        if min_dist2 < cutoff_dist:
                            plt.scatter(tracklets[min_idx][:,0],tracklets[min_idx][:,1], color = (0,0.7,0.7))
                            plt.title("Minimum mean distance: {:.1f}ft,{:.1f}ft".format(min_dist,min_dist2))
                        plt.show()
                        
                        
                        
                        
                        
                        
                ### Now there are initial scores for every tracklet pair
                # we next repeatedly select the best pair of tracklets and merge them, recomputing scores         
                        
        
                break
             
                FINAL = False
                while len(tracklets) > 0 :
                       if len(tracklets) % 10 == 0:
                           durations = [t[-1,0] - t[0,0] for t in tracklets]
                           mean_duration = sum(durations) / len(durations)
                           print("{} Finished, {} Unfinished tracklets with mean duration: {:.1f}s".format(len(tracklets_complete),len(tracklets),mean_duration))
                    
                       ## A disgusting masking operation that should make you queasy
                       # update alignment matrices according to tracklet length logic
                       # if tracklet i is too short, but tracklet j is not, replace tracklet i score with tracklet j backwards score   align_x[i,j] <- align_xb[i,j]
                       # if tracklet j is too short but tracklet i is not, replace tracklet j backwards score with tracklet i score    align_xb[i,j] <- align_x[i,j]       
                       tracklet_lengths = torch.tensor([t.shape[0] for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))
                       mask_ilong = torch.where(tracklet_lengths > min_regression_length,1,0) # 1 where element i is long, 0 otherwise
                       mask_jlong = torch.where(tracklet_lengths.transpose(1,0) > min_regression_length,1,0) # 1 where element j is long, 0 otherwise
                
                       align_x = align_x*mask_ilong + (1-mask_ilong)*(mask_jlong*align_xb + (1-mask_jlong)*big_number)
                       align_y = align_y*mask_ilong + (1-mask_ilong)*(mask_jlong*align_yb + (1-mask_jlong)*big_number)
                
                       align_xb = align_xb*mask_jlong + (1-mask_jlong)*(mask_ilong*align_x + (1-mask_ilong)*big_number)
                       align_yb = align_yb*mask_jlong + (1-mask_jlong)*(mask_ilong*align_y + (1-mask_ilong)*big_number)
                
                
                
                
                
                
                       # compute aggregate scores as a combo of above elements
                       scores = (torch.sqrt(align_x**2 + align_y**2)  + torch.sqrt(align_xb**2 + align_yb**2)) /2
                    
                       ### select the best-scoring association - flatten these two tracklets into a single tracklet
                       # i = merged idx
                       # j = deleted idx
                       best_idx = torch.argmin(scores)
                       best_score = torch.min(scores)
                       keep = [_ for _ in range(len(tracklets))]
        
                       I_COMPLETE = False
                       if best_score > cutoff_dist: 
                           print("WARNING: best score {:.1f} > cutoff distance {}. Consider Terminating".format(best_score,cutoff_dist))
                           
                           if FINAL:
                               break
                           else:
                               FINAL = True
                               
                               if iteration > 12:
                                   # remove all tracklets that are more or less entirely subcontained within another tracklet
                                   durations = (max_t.transpose(1,0) - min_t - t_threshold)[:,0]
                                   
                                   keep = torch.where(durations > min_duration,1,0).nonzero().squeeze(1)
                                   
                                   tracklets = [tracklets[k] for k in keep]
                               
                       if not FINAL:
                           i,j = [best_idx // len(tracklets) , best_idx % len(tracklets)]
                           if i >j: # ensure j > i for deletion purposes
                               j1 = j
                               j = i
                               i = j1
                               
                           SHOW = False
                           if SHOW: 
                               color2 = "b"
                               if FINAL: color2 = "r"
                        
                               plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
                               plt.scatter(tracklets[j][:,0],tracklets[j][:,1], c = color2)
                               plt.title(          "X:{:.1f}ft, Y:{:.1f}ft".format((align_x[i,j]+align_xb[i,j])/2   ,(align_y[i,j]+align_yb[i,j])/2    ))
                               
                               # plot the forward and backward regression lines
                               t1  = tracklets[i][-reg_keep:,0].unsqueeze(1)
                               y1  = tracklets[i][-reg_keep:,1:3]
                               reg = LinearRegression().fit(t1,y1)
                               
                               t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
                               y2  = tracklets[j][:reg_keep,1:3]
                               regb = LinearRegression().fit(t2,y2)
                               
                               t_trend = np.array([[tracklets[i][-1,0]],[tracklets[j][0,0]]])
                               y_trend = reg.predict(t_trend)
                               plt.plot(t_trend,y_trend[:,0],":",c = "k")
                               
                               y_trend2 = regb.predict(t_trend)
                               plt.plot(t_trend,y_trend2[:,0],":",color = (0.5,0.5,0.5))
                
                               
                               plt.show()
                           
                            
                           # combine two tracklet arrays
                           tracklets[i] = torch.cat((tracklets[i],tracklets[j]),dim = 0)
                           
                           # sort by time
                           tracklets[i]  = tracklets[i][tracklets[i][:,0].sort()[1]] 
                           
                           # remove larger idx from tracklets
                           del tracklets[j]
                          
                           
                
                           
                           
                           
                     
                           ## check whether the tracklet has a valid source and sink - if so remove it from further calculation
                           if torch.min(tracklets[i][:,1]) < start_x + x_margin or torch.max(tracklets[i][:,0]) > end_time - t_margin:
                               if torch.max(tracklets[i][:,1]) > end_x  - x_margin or  torch.min(tracklets[i][:,0]) < start_time + t_margin:
                                   tracklets_complete.append(tracklets[i])
                                   del tracklets[i]
                                   I_COMPLETE = True
                                   print("Tracklet {} added to finished queue".format(i))
                                   
                           # update all score matrices t_intersection, x_intersection, y_intersection, intersection, x_align, y_align
                           # if tracklet i was removed, all we need to do is remove row and column i and j
                           # otherwise, we need to remove column and row j, and update column and row i
                           
                           keep.remove(j)
                           if I_COMPLETE:
                               keep.remove(i)
                       
                       # remove old matrix entries
                       max_x = max_x[keep,:][:,keep]
                       min_x = min_x[keep,:][:,keep]
                       maxx_int = maxx_int[keep,:][:,keep]
                       minx_int = minx_int[keep,:][:,keep]
                       x_intersection = x_intersection[keep,:][:,keep]
                       
                       max_y = max_y[keep,:][:,keep]
                       min_y = min_y[keep,:][:,keep]
                       maxy_int = maxy_int[keep,:][:,keep]
                       miny_int = miny_int[keep,:][:,keep]
                       y_intersection = y_intersection[keep,:][:,keep]
                       
                       max_t = max_t[keep,:][:,keep]
                       min_t = min_t[keep,:][:,keep]
                       maxt_int = maxt_int[keep,:][:,keep]
                       mint_int = mint_int[keep,:][:,keep]
                       t_intersection = t_intersection[keep,:][:,keep]
                       
                       intersection   = intersection[keep,:][:,keep]
                       align_x = align_x[keep,:][:,keep]
                       align_y = align_y[keep,:][:,keep]
                       
                       align_xb = align_xb[keep,:][:,keep]
                       align_yb = align_yb[keep,:][:,keep]
                       
                       
                       if not I_COMPLETE and not FINAL: # in this case we need to update each row and column i to reflect new detections added to it
                           max_t[:,i] = torch.max(tracklets[i][:,0]) + 0.5* t_threshold
                           min_t[i,:] = torch.min(tracklets[i][:,0]) - 0.5* t_threshold
                           mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
                           maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
                           zeros = torch.zeros(mint_int.shape,dtype=float)
                           t_intersection = torch.max(zeros, maxt_int-mint_int)
                           t_order = torch.where(max_t.transpose(1,0) - max_t < 0, 1,0)
            
            
                           max_x[:,i] = torch.max(tracklets[i][:,1]) + 0.5* x_threshold
                           min_x[i,:] = torch.min(tracklets[i][:,1]) - 0.5* x_threshold
                           minx_int = torch.max(torch.stack((min_x,min_x.transpose(1,0)),dim = -1),dim = -1)[0]
                           maxx_int = torch.min(torch.stack((max_x,max_x.transpose(1,0)),dim = -1),dim = -1)[0]
                           zeros = torch.zeros(minx_int.shape,dtype=float)
                           x_intersection = torch.max(zeros, maxx_int-minx_int)
                           
                           max_y[:,i] = torch.max(tracklets[i][:,2]) + 0.5* y_threshold
                           min_y[i,:] = torch.min(tracklets[i][:,2]) - 0.5* y_threshold
                           miny_int = torch.max(torch.stack((min_y,min_y.transpose(1,0)),dim = -1),dim = -1)[0]
                           maxy_int = torch.min(torch.stack((max_y,max_y.transpose(1,0)),dim = -1),dim = -1)[0]
                           zeros = torch.zeros(miny_int.shape,dtype=float)
                           y_intersection = torch.max(zeros, maxy_int-miny_int)
                           
                           intersection = t_order *  torch.where(t_intersection > 0,1,0) * torch.where(t_intersection < t_threshold,1,0) * torch.where(x_intersection > 0,1,0)  *  torch.where(y_intersection > 0,1,0) # * d_intersection
                           #torch.where(x_intersection < x_threshold,1,0) * 
                           # zero center diagonal
                           intersection = intersection * (1- torch.eye(intersection.shape[0]))
                           
            
                           # now we need to update align_x and align_y
                           #if tracklets[i].shape[0] < min_regression_length: continue
                           
                       # fit linear regressor
                           t1  = tracklets[i][-reg_keep:,0].unsqueeze(1)
                           y1  = tracklets[i][-reg_keep:,1:3]
                           reg = LinearRegression().fit(t1,y1)
                           
                           t1b = tracklets[i][:reg_keep,0].unsqueeze(1)
                           y1b = tracklets[i][:reg_keep,1:3]
                           regb = LinearRegression().fit(t1b,y1b)
            
                           
                           for j in range(len(tracklets)):
                               if intersection[i,j] == 1:
                                   
                                   # get first bit of data
                                   t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
                                   y2  = tracklets[j][:reg_keep,1:3]
                                   
                                   pred = reg.predict(t2)
                                   diff = np.abs(pred - y2.data.numpy())
                                   
                                   #assign
                                   mdx = diff[:,0].mean()
                                   mdy = diff[:,1].mean()
                                   align_x[i,j] = mdx
                                   align_y[i,j] = mdy
                                   
                                   #threshold
                                   if mdx > x_threshold: align_x[i,j] = 10000
                                   if mdy > y_threshold:_length: align_y[i,j] = 10000
                                   
                                   # if tracklets[i].shape[0] < min_regression_length: 
                                   #     if tracklets[j].shape[0] > min_regression_length:
                                   #         align_x[i,j] = align_xb[i,j]
                                   #         align_y[i,j] = align_yb[i,j]
                                   #     else:
                                   #         align_x[i,j] = big_number
                                   #         align_y[i,j] = big_number
                                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   # backwards regression update
                                   regb = LinearRegression().fit(t2,y2)
                                   pred = regb.predict(t1)
                                   diff = np.abs(pred - y1.data.numpy())
                                   
                                   # assign
                                   mdx = diff[:,0].mean()
                                   mdy = diff[:,1].mean()
                                   align_xb[i,j] = mdx
                                   align_yb[i,j] = mdy
                                   
                                   #threshold
                                   if mdx > x_threshold or tracklets[j].shape[0] < min_regression_length: align_xb[i,j] = 10000
                                   if mdy > y_threshold or tracklets[j].shape[0] < min_regression_length: align_yb[i,j] = 10000
                                   
                                   # if tracklets[j].shape[0] < min_regression_length: 
                                   #     if tracklets[i].shape[0] > min_regression_length:
                                   #         align_xb[i,j] = align_x[i,j]
                                   #         align_yb[i,j] = align_y[i,j]
                                   #     else:
                                   #         align_xb[i,j] = big_number
                                   #         align_yb[i,j] = big_number
                                   
                                    
                                   
                                    
                                   
                                   
                               if intersection[j,i] == 1:
                                    
                                    # get first bit of data
                                    t2  = tracklets[j][-reg_keep:,0].unsqueeze(1)
                                    y2  = tracklets[j][-reg_keep:,1:3]
                                    regj = LinearRegression().fit(t2,y2)
                                    
                                    pred = regj.predict(t1b)
                                    diff = np.abs(pred - y1b.data.numpy())
                                   
                                    # assign
                                    mdx = diff[:,0].mean()
                                    mdy = diff[:,1].mean()
                                    align_x[j,i] = mdx
                                    align_y[j,i] = mdy
                                    
                                    #threshold
                                    if mdx > x_threshold or tracklets[j].shape[0] < min_regression_length: align_x[j,i] = 10000
                                    if mdy > y_threshold or tracklets[j].shape[0] < min_regression_length: align_y[j,i] = 10000
                                    
                                    # if tracklets[j].shape[0] < min_regression_length: 
                                    #     if tracklets[i].shape[0] > min_regression_length:
                                    #         align_x[j,i] = align_xb[j,i]
                                    #         align_y[j,i] = align_yb[j,i]
                                    #     else:
                                    #         align_x[j,i] = big_number
                                    #         align_y[j,i] = big_number
                                            
                                            
                                            
                                            
                                            
                    
                                    # update backwards regression too
                                    pred = regb.predict(t2)
                                    diff = np.abs(pred - y2.data.numpy())
                                    
                                    #assign
                                    mdx = diff[:,0].mean()
                                    mdy = diff[:,1].mean()
                                    align_xb[j,i] = mdx
                                    align_yb[j,i] = mdy
                                    
                                    #threshold
                                    if mdx > x_threshold or tracklets[i].shape[0] < min_regression_length: align_xb[j,i] = 10000
                                    if mdy > y_threshold or tracklets[i].shape[0] < min_regression_length: align_yb[j,i] = 10000
                                    
                                    # if tracklets[i].shape[0] < min_regression_length: 
                                    #     if tracklets[j].shape[0] > min_regression_length:
                                    #         align_xb[j,i] = align_x[j,i]
                                    #         align_yb[j,i] = align_y[j,i]
                                    #     else:
                                    #         align_xb[j,i] = big_number
                                    #         align_yb[j,i] = big_number
                                    
                    
                               # ensure no matches beyond threshold
                               #align_x = torch.where(align_x > x_threshold,10000,align_x)
                               #align_y = torch.where(align_y > y_threshold,10000,align_y)
                            
                 # intersection score - we could use a rasterized roadway map - only one vehicle can occupy each 1 foot by 1 foot grid cell for a timestep - 1 foot grid by 10 Hz = 86 GB ...
                 # initially, every trajectory is painted into this
            
                 # alternatively - for every pair where at least one tracklet is physically and temporally proximal
                  # determine whether a linear interpolation between that pair intersects this trajectory
                  # if we do this, then when two tracklets are combined, their intersection score is an OR operation - if either tracklet A or B could not be paired with tracklet C because of an intersection, then AB cannot be paired with C\
                
                if False:
                    plt.figure(figsize = (20,15))
                    for tidx,t in enumerate(tracklets):
                        if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
                            plt.plot(t[:,0],t[:,1],c = colors[tidx],linewidth = 4)
                            plt.text(t[0,0],t[0,1],tidx)
                    for tidx,t in enumerate(tracklets_complete):
                        if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
                            plt.plot(t[:,0],t[:,1],c = "b",linewidth = 4)
                    plt.title("Post clustering lane {} after pass {}".format(lane,iteration))
                    plt.savefig("im/{}.png".format(str(iteration+1).zfill(3)))
                    plt.show()
            
                
        #%% Phase 3 - tracklet completion using mean speed field   
            t_buffer = 5
            
            msf_dict = generate_msf(tracklets+tracklets_complete,
                         start_time,end_time,
                         start_x,end_x,
                         grid_t=2,grid_x = 50,kwidth = 9)
            
            # for i in range(100):
            #     virtual = get_msf_points(tracklets[i],msf_dict)
            #     plot_tracklet(tracklets,i,virtual = virtual)
            
            x_threshold           = x_thresholds[-1]                  # tracklets beyond this distance apart from one another are not considered
            y_threshold           = y_threshold = y_thresholds[-1]    # tracklets beyond this distance apart from one another are not considered
            t_threshold           = t_thresholds[-1]                  # tracklets beyond this duration apart from one another are not considered
            reg_keep              = reg_keeps[-1]                     # the last _ points are used to fit a linear regression and the first _ points from the next tracklet are scored
            min_regression_length = min_regression_lengths[-1]        # tracklets less than _ in length are not used to fit a linear regression
            cutoff_dist           = cutoff_dists[-1]        
                
            
            # get t overlaps for all tracklets
            start = time.time()
            max_t = torch.tensor([torch.max(t[:,0]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*t_threshold
            min_t = torch.tensor([torch.min(t[:,0]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*t_threshold
               
            mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
            maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
            zeros = torch.zeros(mint_int.shape,dtype=float)
            t_intersection = torch.max(zeros, maxt_int-mint_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
            
            # ensure t1 starts before t2
            t_order = torch.where(max_t.transpose(1,0) - max_t <+ 0, 1,0)
            
            
            # get x overlaps for all tracklets
            max_x = torch.tensor([torch.max(t[:,1]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*x_threshold
            min_x = torch.tensor([torch.min(t[:,1]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*x_threshold
               
            minx_int = torch.max(torch.stack((min_x,min_x.transpose(1,0)),dim = -1),dim = -1)[0]
            maxx_int = torch.min(torch.stack((max_x,max_x.transpose(1,0)),dim = -1),dim = -1)[0]
            zeros = torch.zeros(minx_int.shape,dtype=float)
            x_intersection = torch.max(zeros, maxx_int-minx_int)  # if 0, these two tracklets are not within x_threshold of one another (even disregarding time matching)
            
            # get y overlaps for all tracklets
            max_y = torch.tensor([torch.max(t[:,2]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))  + 0.5*y_threshold
            min_y = torch.tensor([torch.min(t[:,2]) for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))  - 0.5*y_threshold
               
            miny_int = torch.max(torch.stack((min_y,min_y.transpose(1,0)),dim = -1),dim = -1)[0]
            maxy_int = torch.min(torch.stack((max_y,max_y.transpose(1,0)),dim = -1),dim = -1)[0]
            zeros = torch.zeros(miny_int.shape,dtype=float)
            y_intersection = torch.max(zeros, maxy_int-miny_int)  # if 0, these two tracklets are not within y_threshold of one another (even disregarding time matching)
            
            
            #direction
            # direction = torch.tensor([torch.sign(t[0,2]) for t in tracklets]).unsqueeze(0).expand(len(tracklets),len(tracklets))
            # d_intersection = torch.where(direction * direction.transpose(1,0) == 1, 1,0)
            
            intersection = t_order * torch.where(t_intersection > 0,1,0) * torch.where(t_intersection < (t_threshold+t_buffer),1,0) * torch.where(x_intersection > 0,1,0)   * torch.where(y_intersection > 0,1,0) # * d_intersection
            #* torch.where(x_intersection < x_threshold,1,0)
            # zero center diagonal
            intersection = intersection * (1- torch.eye(intersection.shape[0]))
            
            
            
            
            align_x = torch.zeros(intersection.shape) + big_number
            SHOW = False
            # generate initial virtual trajectory scores
            for i in range(len(tracklets)):
                if SHOW: plt.figure()
                
                # get virtual points
                virtual = get_msf_points(tracklets[i],msf_dict)
                
                if SHOW:
                    plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
                    plt.plot(virtual[:,0],virtual[:,1],":",color = (0.2,0.2,0.2))
                
               
                # compare to each candidate match
                for j in range(len(tracklets)):
                    if intersection[i,j] == 1:
                        
                       # nix this # for each point in tracklet [j] within some threshold of the times within virtual, find closest time point, compute x and y distance, and add to score
                       # lets do it incredibly crudely, find first point of tracklet j, find closest virtual trajectory point by time, compute offset
                       jx = tracklets[j][0,1]
                       jt = tracklets[j][0,0]
                       match_idx = torch.argmin(torch.abs(virtual[:,0] - jt))
                       align_x[i,j] = torch.abs(jx - virtual[match_idx,1])
                    
                       if SHOW: 
                            plt.scatter(tracklets[j][:,0],tracklets[j][:,1], color = (0.8,0.8,0))
        
                        
                # find min alignment error
                if SHOW:
                    min_idx = torch.argmin(align_x[i]**2 + align_y[i]**2)
                    min_dist = torch.sqrt(align_x[i,min_idx]**2 + align_y[i,min_idx]**2)
                    if min_dist < cutoff_dist:
                        plt.scatter(tracklets[min_idx][:,0],tracklets[min_idx][:,1], c = "b")
                        plt.title("Minimum mean distance: {:.1f}ft".format(min_dist))
                        plt.show()
                 
                
                 
                
            
            
            
            
            
            
            
            
            
            
           
            
            
        
            FINAL = False
            while len(tracklets) > 0 :
                   if len(tracklets) % 10 == 0:
                       durations = [t[-1,0] - t[0,0] for t in tracklets]
                       mean_duration = sum(durations) / len(durations)
                       print("{} Finished, {} Unfinished tracklets with mean duration: {:.1f}s".format(len(tracklets_complete),len(tracklets),mean_duration))
                
                   ## A disgusting masking operation that should make you queasy
                   # update alignment matrices according to tracklet length logic
                   # if tracklet i is too short, but tracklet j is not, replace tracklet i score with tracklet j backwards score   align_x[i,j] <- align_xb[i,j]
                   # if tracklet j is too short but tracklet i is not, replace tracklet j backwards score with tracklet i score    align_xb[i,j] <- align_x[i,j]       
            
            
            
            
            
            
                   # compute aggregate scores as a combo of above elements
                   scores = torch.sqrt(align_x**2 + align_y**2) 
                
                   ### select the best-scoring association - flatten these two tracklets into a single tracklet
                   # i = merged idx
                   # j = deleted idx
                   best_idx = torch.argmin(scores)
                   best_score = torch.min(scores)
                   keep = [_ for _ in range(len(tracklets))]
        
                   if best_score > cutoff_dist: 
                       print("WARNING: best score {:.1f} > cutoff distance {}. Consider Terminating".format(best_score,cutoff_dist))
                       
                       if FINAL:
                           break
                       else:
                           FINAL = True
                           
                        
                           
                   if not FINAL:
                       i,j = [best_idx // len(tracklets) , best_idx % len(tracklets)]
                       if i > j:
                           j1 = i
                           i = j
                           j = j1
                           
                       SHOW = False
                       if SHOW: 
                           color2 = "b"
                           if FINAL: color2 = "r"
                    
                           plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
                           plt.scatter(tracklets[j][:,0],tracklets[j][:,1], c = color2)
                           plt.title(          "X:{:.1f}ft, Y:{:.1f}ft".format(align_x[i,j],align_y[i,j]))
                           
                           plt.show()
                       
                        
                       # combine two tracklet arrays
                       tracklets[i] = torch.cat((tracklets[i],tracklets[j]),dim = 0)
                       
                       # sort by time
                       tracklets[i]  = tracklets[i][tracklets[i][:,0].sort()[1]] 
                       
                       # remove larger idx from tracklets
                       del tracklets[j]
                      
                       
            
                       
                       
                       
                 
                       ## check whether the tracklet has a valid source and sink - if so remove it from further calculation
                       I_COMPLETE = False
                       if torch.min(tracklets[i][:,1]) < start_x + x_margin or torch.max(tracklets[i][:,0]) > end_time - t_margin:
                           if torch.max(tracklets[i][:,1]) > end_x  - x_margin or  torch.min(tracklets[i][:,0]) < start_time + t_margin:
                               tracklets_complete.append(tracklets[i])
                               del tracklets[i]
                               I_COMPLETE = True
                               print("Tracklet {} added to finished queue".format(i))
                               
                       # update all score matrices t_intersection, x_intersection, y_intersection, intersection, x_align, y_align
                       # if tracklet i was removed, all we need to do is remove row and column i and j
                       # otherwise, we need to remove column and row j, and update column and row i
                       
                       keep.remove(j)
                       if I_COMPLETE:
                           keep.remove(i)
                   
                   # remove old matrix entries
                   max_x = max_x[keep,:][:,keep]
                   min_x = min_x[keep,:][:,keep]
                   maxx_int = maxx_int[keep,:][:,keep]
                   minx_int = minx_int[keep,:][:,keep]
                   x_intersection = x_intersection[keep,:][:,keep]
                   
                   max_y = max_y[keep,:][:,keep]
                   min_y = min_y[keep,:][:,keep]
                   maxy_int = maxy_int[keep,:][:,keep]
                   miny_int = miny_int[keep,:][:,keep]
                   y_intersection = y_intersection[keep,:][:,keep]
                   
                   max_t = max_t[keep,:][:,keep]
                   min_t = min_t[keep,:][:,keep]
                   maxt_int = maxt_int[keep,:][:,keep]
                   mint_int = mint_int[keep,:][:,keep]
                   t_intersection = t_intersection[keep,:][:,keep]
                   
                   intersection   = intersection[keep,:][:,keep]
                   align_x = align_x[keep,:][:,keep]
                   align_y = align_y[keep,:][:,keep]
                   
                  
                   
                   if not I_COMPLETE and not FINAL: # in this case we need to update each row and column i to reflect new detections added to it
                       max_t[:,i] = torch.max(tracklets[i][:,0]) + 0.5* t_threshold
                       min_t[i,:] = torch.min(tracklets[i][:,0]) - 0.5* t_threshold
                       mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
                       maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
                       zeros = torch.zeros(mint_int.shape,dtype=float)
                       t_intersection = torch.max(zeros, maxt_int-mint_int)
                       t_order = torch.where(max_t.transpose(1,0) - max_t < 0, 1,0)
        
        
                       max_x[:,i] = torch.max(tracklets[i][:,1]) + 0.5* x_threshold
                       min_x[i,:] = torch.min(tracklets[i][:,1]) - 0.5* x_threshold
                       minx_int = torch.max(torch.stack((min_x,min_x.transpose(1,0)),dim = -1),dim = -1)[0]
                       maxx_int = torch.min(torch.stack((max_x,max_x.transpose(1,0)),dim = -1),dim = -1)[0]
                       zeros = torch.zeros(minx_int.shape,dtype=float)
                       x_intersection = torch.max(zeros, maxx_int-minx_int)
                       
                       max_y[:,i] = torch.max(tracklets[i][:,2]) + 0.5* y_threshold
                       min_y[i,:] = torch.min(tracklets[i][:,2]) - 0.5* y_threshold
                       miny_int = torch.max(torch.stack((min_y,min_y.transpose(1,0)),dim = -1),dim = -1)[0]
                       maxy_int = torch.min(torch.stack((max_y,max_y.transpose(1,0)),dim = -1),dim = -1)[0]
                       zeros = torch.zeros(miny_int.shape,dtype=float)
                       y_intersection = torch.max(zeros, maxy_int-miny_int)
                       
                       intersection = t_order *  torch.where(t_intersection > 0,1,0) * torch.where(t_intersection < (t_threshold+t_buffer),1,0) * torch.where(x_intersection > 0,1,0)  *  torch.where(y_intersection > 0,1,0) # * d_intersection
                       #torch.where(x_intersection < x_threshold,1,0) * 
                       # zero center diagonal
                       intersection = intersection * (1- torch.eye(intersection.shape[0]))
                       
        
                       # now we need to update align_x and align_y
        
        
                       virtual =  get_msf_points(tracklets[i],msf_dict)
                       for j in range(len(tracklets)):
                           if intersection[i,j] == 1:
                               
                              
                               # lets do it incredibly crudely, find first point of tracklet j, find closest virtual trajectory point by time, compute offset
                               jx = tracklets[j][0,1]
                               jt = tracklets[j][0,0]
                               match_idx = torch.argmin(torch.abs(virtual[:,0] - jt))
                               mdx = torch.abs(jx - virtual[match_idx,1])
                               
                               # get first bit of data
                               t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
                               y2  = tracklets[j][:reg_keep,1:3]
                               
                               pred = reg.predict(t2)
                               diff = np.abs(pred - y2.data.numpy())
                               
                               #assign
                               mdy = diff[:,1].mean()
                               align_y[i,j] = mdy
                               align_x[i,j] = mdx
                              
                               #threshold
                               if mdx > x_threshold: align_x[i,j] = 10000
                               if mdy > y_threshold:_length: align_y[i,j] = 10000
                               
                               
                           # if intersection[j,i] == 1:
                                
                           #      # get first bit of data
                           #      t2  = tracklets[j][-reg_keep:,0].unsqueeze(1)
                           #      y2  = tracklets[j][-reg_keep:,1:3]
                           #      regj = LinearRegression().fit(t2,y2)
                                
                           #      pred = regj.predict(t1b)
                           #      diff = np.abs(pred - y1b.data.numpy())
                               
                           #      # assign
                           #      mdx = diff[:,0].mean()
                           #      mdy = diff[:,1].mean()
                           #      align_x[j,i] = mdx
                           #      align_y[j,i] = mdy
                                
                           #      #threshold
                           #      if mdx > x_threshold or tracklets[j].shape[0] < min_regression_length: align_x[j,i] = 10000
                           #      if mdy > y_threshold or tracklets[j].shape[0] < min_regression_length: align_y[j,i] = 10000
                                
                           #      # if tracklets[j].shape[0] < min_regression_length: 
                           #      #     if tracklets[i].shape[0] > min_regression_length:
                           #      #         align_x[j,i] = align_xb[j,i]
                           #      #         align_y[j,i] = align_yb[j,i]
                           #      #     else:
                           #      #         align_x[j,i] = big_number
                           #      #         align_y[j,i] = big_number
                                        
                                        
                                        
                                        
                                        
                
                           #      # update backwards regression too
                           #      pred = regb.predict(t2)
                           #      diff = np.abs(pred - y2.data.numpy())
                                
                           #      #assign
                           #      mdx = diff[:,0].mean()
                           #      mdy = diff[:,1].mean()
                           #      align_xb[j,i] = mdx
                           #      align_yb[j,i] = mdy
                                
                           #      #threshold
                           #      if mdx > x_threshold or tracklets[i].shape[0] < min_regression_length: align_xb[j,i] = 10000
                           #      if mdy > y_threshold or tracklets[i].shape[0] < min_regression_length: align_yb[j,i] = 10000
                                
        
            # save data
            with open(save_file,"wb") as f:
                pickle.dump(tracklets + tracklets_complete,f)
        
            plt.figure(figsize = (20,15))
            for tidx,t in enumerate(tracklets):
                if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
                    plt.plot(t[:,0],t[:,1],c = colors[tidx],linewidth = 4)
                    plt.text(t[0,0],t[0,1],tidx)
            for tidx,t in enumerate(tracklets_complete):
                if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
                    plt.plot(t[:,0],t[:,1],c = "b",linewidth = 4)
            plt.title("Post clustering lane {} after pass {}".format(lane,iteration))
            plt.savefig("im/{}.png".format(str(iteration+1).zfill(3)))
            plt.show()