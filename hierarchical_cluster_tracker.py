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
from scipy.stats import norm

from matplotlib.patches import Rectangle

colors = np.random.rand(10000,3)
#colors[:,2] = 0    


mp.set_sharing_strategy('file_system')



# ok look
# here's the deal - I know this is horrible horrible practice to import shared variables this way but I'm just too lazy to pass yet another data file to each process
zones = { "WB":{
             "source":{
                 "bell":[3300,3400,-120,-65],
                 "hickoryhollow":[6300,6500,-120,-65],
                 "p25":[13300,13550,-100,0],
                 "oldhickory":[18300,18700,-120,-65],
                 "extent":[21900,23000,-100,0]
                 },
             "sink":{
                 "extent":[-1000,0,-100,0],
                 "bell":[4400,4600,-120,-65],
                 "hickoryhollow":[8000,8200,-120,-65],
                 "p25":[13450,13650,-100,0],
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





class Timer:
    
    def __init__(self):
        self.cur_section = None
        self.sections = {}
        self.section_calls = {}
        
        self.start_time= time.time()
        self.split_time = None
        self.synchronize = False
        
    def split(self,section,SYNC = False):

        # store split time up until now in previously active section (cur_section)
        if self.split_time is not None:
            if SYNC and self.synchronize:
                torch.cuda.synchronize()
                
            elapsed = time.time() - self.split_time
            if self.cur_section in self.sections.keys():
                self.sections[self.cur_section] += elapsed
                self.section_calls[self.cur_section] += 1

            else:
                self.sections[self.cur_section] = elapsed
                self.section_calls[self.cur_section] = 1
        # start new split and activate current time
        self.cur_section = section
        self.split_time = time.time()
        
    def bins(self):
        self.sections["total"] = time.time() - self.start_time
        return self.sections
    
    def __repr__(self):
        out = ["{}:{:2f}s/call".format(key,self.sections[key]/self.section_calls[key]) for key in self.sections.keys()]
        return str(out)


#%% Phase 1 functions

# def hierarchical_tree_old(start_x,end_x,start_time,end_time,space_chunk,time_chunk,split_space = True):
#     tree = {"start_time":start_time,
#             "end_time": end_time,
#             "start_x":start_x,
#             "end_x":end_x,
#             "data":None,
#             "children":None}
    
#     if end_time - start_time > time_chunk:
#         middle_time = start_time + (end_time - start_time)//2
#         if end_x - start_x > space_chunk:
#             middle_x = start_x + (end_x - start_x)//2

#             children = [
#                 hierarchical_tree(start_x,middle_x,start_time,middle_time,space_chunk,time_chunk),
#                 hierarchical_tree(start_x,middle_x,middle_time,end_time,space_chunk,time_chunk),
#                 hierarchical_tree(middle_x,end_x,start_time,middle_time,space_chunk,time_chunk),
#                 hierarchical_tree(middle_x,end_x,middle_time,end_time,space_chunk,time_chunk)
#                 ]
#         else:
#             children = [
#                 hierarchical_tree(start_x,end_x,start_time,middle_time,space_chunk,time_chunk),
#                 hierarchical_tree(start_x,end_x,middle_time,end_time,space_chunk,time_chunk) 
#                 ]
        
#     elif end_x - start_x > space_chunk:
#         middle_x = start_x + (end_x - start_x)//2
        
#         children = [
#             hierarchical_tree(start_x,middle_x,start_time,end_time,space_chunk,time_chunk),
#             hierarchical_tree(middle_x,end_x,start_time,end_time,space_chunk,time_chunk)
#             ]
    
#     else:
#         children = []
    
#     tree["children"] = children
#     return tree

def hierarchical_tree(start_x,end_x,start_time,end_time,space_chunk,time_chunk,pass_param_stack,split_space = True):
    tree = {"start_time":start_time,
            "end_time": end_time,
            "start_x":start_x,
            "end_x":end_x,
            "interface_x":None,
            "interface_width_x":None,
            "interface_t": None,
            "interface_width_t":None,
            "data":None,
            "children":None,
            "pass_params":None}
    
    """
    Balanced to split only on space or time dimension on a single split layer
    """
    
    # fill in pass params
    for p in pass_param_stack:
        if p[0] >= end_time - start_time and p[1] >= np.abs(start_x-end_x):
            tree["pass_params"] = p[2]
            break
        
    # if tree["pass_params"] is None:
    #     raise AssertionError( "There are no valid pass_param configurations for at least this block")
        
    
    if (end_time - start_time > time_chunk) and (not split_space or (end_x - start_x) <= space_chunk):
        middle_time = start_time + (end_time - start_time)//2
        
        if tree["pass_params"] is not None and tree["pass_params"]["INTERFACE"]:
            tree["interface_t"] = middle_time
            tree["interface_width_t"] = time_chunk //2  
        
        # if end_x - start_x > space_chunk:
        #     middle_x = start_x + (end_x - start_x)//2

        #     children = [
        #         hierarchical_tree(start_x,middle_x,start_time,middle_time,space_chunk,time_chunk),
        #         hierarchical_tree(start_x,middle_x,middle_time,end_time,space_chunk,time_chunk),
        #         hierarchical_tree(middle_x,end_x,start_time,middle_time,space_chunk,time_chunk),
        #         hierarchical_tree(middle_x,end_x,middle_time,end_time,space_chunk,time_chunk)
        #         ]
        # else:
        children = [
            hierarchical_tree(start_x,end_x,start_time,middle_time,space_chunk,time_chunk,pass_param_stack,split_space = not(split_space)),
            hierarchical_tree(start_x,end_x,middle_time,end_time,space_chunk,time_chunk,pass_param_stack,split_space = not(split_space)) 
            ]
        
    elif end_x - start_x > space_chunk:
        middle_x = start_x + (end_x - start_x)//2
        if tree["pass_params"] is not None and tree["pass_params"]["INTERFACE"]:
            tree["interface_x"] = middle_x
            tree["interface_width_x"] = space_chunk//2
        
        children = [
            hierarchical_tree(start_x,middle_x,start_time,end_time,space_chunk,time_chunk,pass_param_stack,split_space = not(split_space)),
            hierarchical_tree(middle_x,end_x,start_time,end_time,space_chunk,time_chunk,pass_param_stack,split_space = not(split_space))
            ]
    
    else:
        children = []
    
    tree["children"] = children
    return tree

def time_ordered_tree(start_x,end_x,start_time,end_time,time_chunk):
    """
    Split time up into sequential chunks
    """
    
    tree = {"start_time":start_time,
            "end_time": end_time,
            "start_x":start_x,
            "end_x":end_x,
            "data":None,
            "children":None,
            "interface_x":None,
            "interface_width_x":None,
            "interface_t": None,
            "interface_width_t":None
            }
    
    
    if (end_time - start_time > time_chunk):
        middle_time = end_time - time_chunk
        tree["interface_t"] = middle_time
        tree["interface_width_t"] = time_chunk
        # if end_x - start_x > space_chunk:
        #     middle_x = start_x + (end_x - start_x)//2

        #     children = [
        #         hierarchical_tree(start_x,middle_x,start_time,middle_time,space_chunk,time_chunk),
        #         hierarchical_tree(start_x,middle_x,middle_time,end_time,space_chunk,time_chunk),
        #         hierarchical_tree(middle_x,end_x,start_time,middle_time,space_chunk,time_chunk),
        #         hierarchical_tree(middle_x,end_x,middle_time,end_time,space_chunk,time_chunk)
        #         ]
        # else:
        children = [
            time_ordered_tree(start_x,end_x,start_time,middle_time,time_chunk),
            time_ordered_tree(start_x,end_x,middle_time,end_time,time_chunk) 
            ]
    else:
        children = []
    
    tree["children"] = children
    return tree

def get_tree_leaves(tree):
    leaves = []
    
    if len(tree["children"]) == 0:
        leaves.append(tree)
    else:
        for c in tree["children"]:
            leaves += get_tree_leaves(c)
        
    return leaves
    
def flatten_tree(tree,data_dir):
    
    if len(tree["children"]) > 0:
        tree_list = []
        for i in range(len(tree["children"])):
            tree_list += flatten_tree(tree["children"][i],data_dir)
    else:
        tree_list = []
        
    tree["dep"] = []
    for child in tree["children"]:
        path = "{}/tracklets_{}_{}_{}_{}.cpkl".format(data_dir,child["start_time"],child["end_time"],child["start_x"],child["end_x"])
        tree["dep"].append(path)
        
    # for now supress this so there are no dependencies for bottom level nodes
    # if len(tree["dep"]) == 0:
    #     tree["dep"].append( "{}/clusters_{}_{}_{}_{}.npy".format(data_dir,tree["start_time"],tree["end_time"],tree["start_x"],tree["end_x"]))
    
    del tree["children"]
    tree["my_path"] = "{}/tracklets_{}_{}_{}_{}.cpkl".format(data_dir,tree["start_time"],tree["end_time"],tree["start_x"],tree["end_x"])
    
    tree_list.append(tree)
        
    return tree_list
    
    

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

def cluster(det,start_time,end_time,start_x,end_x,direction, delta = 0.3, phi = 0.4,data_dir = "data"):
    def visit(i,id = -1):
        # if this node has an id, return
        # else, assign id, and return list of neighbors
        if ids[i] != -1:
            return []
        else:
            ids[i] = id
            return adj_list[i]
    
    path = "{}/clusters_{}_{}_{}_{}.npy".format(data_dir,start_time,end_time,start_x,end_x)
    if os.path.exists(path): 
        return -1


    # Phase 1 tracklet prep
    
    # select only relevant detections
    if type(det) == np.ndarray:
        det = torch.from_numpy(det)
    
    print("Checkpoint 0")
    time_idxs = torch.where(torch.logical_and(det[:,0] > start_time,det[:,0] < end_time),1,0)
    print("Checkpoint 1")

    space_idxs = torch.where(torch.logical_and(det[:,1] > start_x,det[:,1]<end_x),1,0)
    direction_idxs = torch.where(torch.sign(det[:,2]) == direction,1,0)
    keep_idxs = (time_idxs * space_idxs * direction_idxs).nonzero().squeeze()
    det = det[keep_idxs,:]
    
    ### detections have many overlaps - the first phase groups all detections that overlap sufficiently in space, and that create continuous clusters in space (i.e. iou-based tracker)
    t1 = time.time()
    ids = torch.torch.tensor([i for i in range(det.shape[0])]) # to start every detection is in a distinct cluster
    adj_list = [[] for i in range(det.shape[0])]
    
    for ts in np.arange(start_time,end_time,step = 0.1):
        elapsed = time.time() - t1
        remaining = elapsed/(ts-start_time+0.001) * (end_time - ts)
        print("\r Processing time {:.1f}/{:.1f}s     {:.2f}% done, ({:.1f}s elapsed, {:.1f}s remaining.)   ".format(ts,end_time,(ts-start_time)*100/(end_time-start_time),elapsed,remaining),flush = True, end = "\r")
        
        ## grab the set of detections in ts,ts+delta
        ts_idxs = torch.where(torch.logical_and(det[:,0] > ts,det[:,0] < ts+delta),1,0).nonzero().squeeze()
        ts_det = det[ts_idxs,:]
        max_idx = torch.max(ts_idxs)
        
        if ts_det.shape[0] == 0:
            continue
        first  = torch.clone(ts_det)
        
        ## convert from state for to state-space box rcs box form
        boxes_new = torch.zeros([first.shape[0],4],)
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
            
                

        
       
    ##, now given an adjacency list for each detection, get clusters
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
               
               
               
               
    
    ## resulting will be a set of tracklet clusters (clusters of detections)
    count = ids.unique().shape[0]
    orig_idxs =  torch.torch.tensor([i for i in range(det.shape[0])])
    out = torch.cat((det,orig_idxs.unsqueeze(1),ids.unsqueeze(1)),dim = 1)
    out = out.data.numpy()
    np.save(path,out)
    
    # time x y l w h class confidence original id clustered id
    
    print("\nFinished clustering detections for phase 1. Before:{}, After: {}. {:.1f}s elapsed.".format(det.shape[0],count,time.time() - t1))

#%% Phase 2 functions
def resample(tracklets,tracklets_complete, sigma = 0.1,hz = 10):
    for tidx, tr in enumerate( tracklets):
        # print("resampling tracklet {}".format(tidx))
        
        ts = tr[:,0]
        x = tr[:,1]
        y = tr[:,2]
        
        ts_resampled = ts #torch.arange(start = ts[0],end = ts[-1]+1/hz,step = 1/hz,dtype = ts.dtype)
        x_resampled = torch.zeros(ts_resampled.shape[0],dtype = ts.dtype)
        y_resampled = torch.zeros(ts_resampled.shape[0],dtype = ts.dtype)
        
        
        
        for i in range(len(ts_resampled)):
            normal = norm(loc = ts_resampled[i],scale = sigma)
            
            weights = normal.pdf(ts)
            if sum(weights) == 0:
                normal = norm(loc = ts_resampled[i],scale = sigma*5)
                weights = normal.pdf(ts)
            weights = weights/sum(weights)
            
            x_resampled[i] = (x*weights).sum()
            y_resampled[i] = (y*weights).sum()
            
            
        # TODO - we delete all the channels except t,x,y - add them back later
        tracklets[tidx] = torch.stack([ts_resampled,x_resampled,y_resampled]).transpose(1,0)
    
    for tidx, tr in enumerate(tracklets_complete):
        
        ts = tr[:,0]
        x = tr[:,1]
        y = tr[:,2]
        
        ts_resampled = ts #torch.arange(start = ts[0],end = ts[-1]+1/hz,step = 1/hz,dtype = ts.dtype)
        x_resampled = torch.zeros(ts_resampled.shape[0],dtype = ts.dtype)
        y_resampled = torch.zeros(ts_resampled.shape[0],dtype = ts.dtype)
        
        
        
        
        for i in range(len(ts_resampled)):
            normal = norm(loc = ts_resampled[i],scale = sigma)
            weights = normal.pdf(ts)
            if sum(weights) == 0:
                normal = norm(loc = ts_resampled[i],scale = sigma*5)
                weights = normal.pdf(ts)
            weights = weights/sum(weights)
            
            x_resampled[i] = (x*weights).sum()
            y_resampled[i] = (y*weights).sum()
            
            
        # TODO - we delete all the channels except t,x,y - add them back later
        tracklets_complete[tidx] = torch.stack([ts_resampled,x_resampled,y_resampled]).transpose(1,0)
    
    return tracklets,tracklets_complete
    
def compute_intersections(tracklets,t_threshold,x_threshold,y_threshold, i = None, intersection = None,intersection_other = None,seam_idx = None,j = None):
    
    """
    if intersection and index are passed, update is performed solely on row/column i and the resulting updated intersection is returnedc
    """
    
    if i is None or i >= (len(tracklets)): # second case covers when i was the last tracklet and was added to finished queue
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
        
        intersection = t_order * torch.where(t_intersection > 0,1,0) * torch.where(t_intersection < (t_threshold+2),1,0) * torch.where(x_intersection > 0,1,0)   * torch.where(y_intersection > 0,1,0) # * d_intersection
        #* torch.where(x_intersection < x_threshold,1,0)
        # zero center diagonal
        intersection = intersection * (1- torch.eye(intersection.shape[0]))
        
        
        #print("Intersection computation took {:.2f}s".format(time.time() - start))
        
        # only consider pre-seam -> post-seam matches
        if seam_idx is not None:
            # intersection[:seam_idx,:] = 0
            # intersection[seam_idx:,seam_idx:] = 0
            intersection[:seam_idx,:seam_idx] = 0
            intersection[seam_idx:,seam_idx:] = 0
        
    else:  
        # "load" parameters
        max_t = intersection_other["max_t"]
        min_t = intersection_other["min_t"]
        maxt_int = intersection_other["maxt_int"]
        mint_int = intersection_other["mint_int"]
        t_intersection = intersection_other["t_intersection"]
       
        max_x = intersection_other["max_x"]
        min_x = intersection_other["min_x"]
        maxx_int = intersection_other["maxx_int"]
        minx_int = intersection_other["minx_int"]
        x_intersection = intersection_other["x_intersection"]        
        
        max_y = intersection_other["max_y"]
        min_y = intersection_other["min_y"]
        maxy_int = intersection_other["maxy_int"]
        miny_int = intersection_other["miny_int"]
        y_intersection = intersection_other["y_intersection"]
        
        if j is None:
            
            
            # update row/column
            # max_t[:,i] = torch.max(tracklets[i][:,0]) + 0.5* t_threshold
            # min_t[i,:] = torch.min(tracklets[i][:,0]) - 0.5* t_threshold
            max_t[:,i] = tracklets[i][-1,0] + 0.5* t_threshold
            min_t[i,:] = tracklets[i][0,0] - 0.5* t_threshold
            mint_int = torch.max(torch.stack((min_t,min_t.transpose(1,0)),dim = -1),dim = -1)[0]
            maxt_int = torch.min(torch.stack((max_t,max_t.transpose(1,0)),dim = -1),dim = -1)[0]
            zeros = torch.zeros(mint_int.shape,dtype=float)
            t_intersection = torch.max(zeros, maxt_int-mint_int)
            t_order = torch.where(max_t.transpose(1,0) - max_t < 0, 1,0)
    
    
            max_x[:,i] = max(tracklets[i][0,1],tracklets[i][-1,1]) + 0.5* x_threshold
            min_x[i,:] = min(tracklets[i][0,1],tracklets[i][-1,1]) - 0.5* x_threshold
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
            
            
            
            # only consider pre-seam -> post-seam matches
            if seam_idx is not None:
                # intersection[:seam_idx,:] = 0
                # intersection[seam_idx:,seam_idx:] = 0
                intersection[:seam_idx,:seam_idx] = 0
                intersection[seam_idx:,seam_idx:] = 0
                
        # intersection of new tracklet is the union of the intersections of the two old components - though I think this should probably deal with the interesections that now fall within the tracklet
        elif j is not None:
            intersection[i,:] = torch.clamp(intersection[i,:] + intersection[j,:],min = 0,max = 1)
            intersection[:,i] = torch.clamp(intersection[:,i] + intersection[:,j],min = 0,max = 1)
    
    intersection_other = {
        "max_t":max_t,
        "min_t":min_t,
        "maxt_int":maxt_int,
        "mint_int":mint_int,
        "t_intersection":t_intersection,
        "max_x":max_x,
        "min_x":min_x,
        "minx_int":minx_int,
        "maxx_int":maxx_int,
        "x_intersection":x_intersection,
        "max_y":max_y,
        "min_y":min_y,
        "miny_int":miny_int,
        "maxy_int":maxy_int,
        "y_intersection":y_intersection,
        }
    return intersection,intersection_other
    
def compute_raster_pos(tracklets, start_time,end_time,start_x,end_x,hz = 0.2, i = None, raster_pos =  None):
    
    
    raster_times = torch.arange(start_time,end_time, step = hz)
    
    if raster_pos is None or i is None:  
            raster_pos = torch.zeros([len(tracklets),raster_times.shape[0],2]) + torch.nan
            i_list = list(range(len(tracklets)))
    else:
        i_list = [i]
    
    for i in i_list:
        #if i%100 == 0: print("Getting raster positions for tracklet {}".format(i))
        t = tracklets[i]
        tidx = 0
        
        m1 = max(0,int((t[0,0] - start_time)//hz - 1))
        m2 = min(raster_times.shape[0],int((t[-1,0] - start_time)//hz + 1))
        for ridx in range(m1,m2):
            rt = raster_times[ridx]
            if rt < t[0,0]:
                continue
            elif rt > t[-1,0] :
                continue
            else:
                while t[tidx,0] < rt:
                    tidx+= 1
                    
                t1 = t[tidx-1,0]
                t2 = t[tidx  ,0]
                x1 = t[tidx-1,1]
                x2 = t[tidx  ,1]
                y1 = t[tidx-1,2]
                y2 = t[tidx,  2]
                
                r2 = (rt-t1) / (t2-t1)
                r1 = 1-r2
                
                x_rt = x1*r1  + x2*r2
                y_rt = y1*r1  + y2*r2
                raster_pos[i,ridx,0] = x_rt
                raster_pos[i,ridx,1] = y_rt
        
        
        
    return raster_pos
                
                
        # interpolate position at raster_times[ridx]
    
    
def compute_scores_linear(tracklets,
                           intersection,
                           params,
                           iteration,
                           align_x,
                           align_y,
                           align_xb,
                           align_yb,
                           i = None
                           ):
    # load params arggh what a dumb solution
     t_threshold = params["t_thresholds"][iteration]
     x_threshold = params["x_thresholds"][iteration]
     y_threshold = params["y_thresholds"][iteration]
     min_regression_length = params["min_regression_lengths"][iteration]
     reg_keep = params["reg_keeps"][iteration]
     cutoff_dist = params["cutoff_dists"][iteration]
     big_number = params["big_number"]
     
     SHOW = False
     
     if i is None:
         i_list = list(range(len(tracklets)))
         
         mask1 = torch.where(align_x == big_number,1,0) # these are the only elements that have changed since last iteration
         mask = (mask1*intersection).nonzero()
         i_list = torch.cat((mask[:,0], mask[:,1]))
         j_list = torch.cat((mask[:,1], mask[:,0]))
         
     else:
         temp_j = intersection[i,:].nonzero().squeeze(1)
         temp_i = torch.tensor([i for _ in range(len(temp_j))])
         
         i_list = torch.cat((temp_i,temp_j))
         j_list = torch.cat((temp_j,temp_i))
         
     prev_i = -1
     
     
     for list_idx,i in enumerate(i_list):
         
         if tracklets[i].shape[0] < min_regression_length: continue
            
         if i != prev_i: #reuse regression if possible
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
         
         #if i%100 == 0: print("On tracklet {} of {}".format(i,len(tracklets)))
         
         j = j_list[list_idx]
     
         if intersection[i,j] == 1:
            
            #if align_x[i,j] != big_number: continue # if we already computed a score for this pair, that score is the same
            
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
             
     # mask short
     if False:
         for idx,i in enumerate(i_list):
             duration_i = tracklets[i][-1,0] - tracklets[i][0,0]
             if duration_i < min_regression_length:
                 j = j_list[idx]
                 duration_j = tracklets[j][-1,0] - tracklets[j][0,0]
                 if duration_j > min_regression_length:
                     
                    if intersection[i,j] == 1:
                         align_x[i,j] = align_xb[i,j]
                         align_y[i,j] = align_yb[i,j]
                     
                    if intersection[j,i] == 1:
                        align_xb[j,i] = align_x[j,i]
                        align_yb[j,i] = align_y[j,i]
                    
      
     return align_x,align_y,align_xb,align_yb


def compute_scores_msf(tracklets,
                        intersection,
                        params,
                        iteration,
                        align_x,
                        align_y,
                        msf_dict,
                        i = None
                        ):
    
     # load params arggh what a dumb solution
     t_threshold = params["t_thresholds"][iteration]
     x_threshold = params["x_thresholds"][iteration]
     y_threshold = params["y_thresholds"][iteration]
     min_regression_length = params["min_regression_lengths"][iteration]
     reg_keep = params["reg_keeps"][iteration]
     cutoff_dist = params["cutoff_dists"][iteration]
     big_number = params["big_number"]
     
     SHOW = False
     
     if i is None:
         i_list = list(range(len(tracklets)))
     else:
         i_list = [i]
     
     for i in i_list:
         if SHOW: plt.figure()
         
         # get virtual points
         virtual = get_msf_points(tracklets[i],msf_dict)
         
         if SHOW:
             plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
             plt.plot(virtual[:,0],virtual[:,1],":",color = (0.2,0.2,0.2))
         
         # linear regressor - for y-coordinates
         t1 = tracklets[i][:reg_keep,0].unsqueeze(1)
         y1 = tracklets[i][:reg_keep,1:3]
         reg = LinearRegression().fit(t1,y1)
         
         # compare to each candidate match
         for j in range(len(tracklets)):
             if intersection[i,j] == 1 and align_x[i,j] == big_number:
                 
                # nix this # for each point in tracklet [j] within some threshold of the times within virtual, find closest time point, compute x and y distance, and add to score
                # lets do it incredibly crudely, find first point of tracklet j, find closest virtual trajectory point by time, compute offset
                jx = tracklets[j][0,1]
                jt = tracklets[j][0,0]
                match_idx = torch.argmin(torch.abs(virtual[:,0] - jt))
                align_x[i,j] = torch.abs(jx - virtual[match_idx,1])
             
                # get first bit of data
                t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
                y2  = tracklets[j][:reg_keep,1:3]
                
                pred = reg.predict(t2)
                diff = np.abs(pred - y2.data.numpy())
                
                #assign
                mdy = diff[:,1].mean()
                align_y[i,j] = mdy

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

     return align_x,align_y

def compute_scores_msf2(tracklets,
                        intersection,
                        params,
                        iteration,
                        align_x,
                        align_y,
                        msf_dict,
                        i = None
                        ):
    
     # load params arggh what a dumb solution
     t_threshold = params["t_thresholds"][iteration]
     x_threshold = params["x_thresholds"][iteration]
     y_threshold = params["y_thresholds"][iteration]
     min_regression_length = params["min_regression_lengths"][iteration]
     reg_keep = params["reg_keeps"][iteration]
     cutoff_dist = params["cutoff_dists"][iteration]
     big_number = params["big_number"]
     
     SHOW = False
     virtuals = None
     
     if i is None:
         i_list = list(range(len(tracklets)))
         
         mask1 = torch.where(align_x == big_number,1,0) # these are the only elements that have changed since last iteration
         mask = (mask1*intersection).nonzero()
         i_list = mask[:,0]
         j_list = mask[:,1]
         
         # precompute msf for ALLL - this may be slower since we don't downselect only relevant tracklets - though we could
         virtuals = get_msf_points_batched(tracklets, msf_dict)
         
     else:
         j_list = intersection[i,:].nonzero().squeeze(1)
         i_list = torch.tensor([i for _ in range(len(j_list))])
         
         
         
     
         
     prev_i = -1
     for list_idx, i in enumerate(i_list):
         if SHOW: plt.figure()
         
         # get virtual points if i has changed
         if virtuals is not None:
             virtual = virtuals[i]
         elif i != prev_i:
             virtual = get_msf_points(tracklets[i],msf_dict)
         
         if SHOW:
             plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
             plt.plot(virtual[:,0],virtual[:,1],":",color = (0.2,0.2,0.2))
         
         # linear regressor - for y-coordinates
         t1 = tracklets[i][:reg_keep,0].unsqueeze(1)
         y1 = tracklets[i][:reg_keep,1:3]
         reg = LinearRegression().fit(t1,y1)
         
         # compare to each candidate match
         #for j in range(len(tracklets)):
         j = j_list[list_idx]
         if intersection[i,j] == 1 and align_x[i,j] == big_number:
            
           # nix this # for each point in tracklet [j] within some threshold of the times within virtual, find closest time point, compute x and y distance, and add to score
           # lets do it incredibly crudely, find first point of tracklet j, find closest virtual trajectory point by time, compute offset
           jx = tracklets[j][0,1]
           jt = tracklets[j][0,0]
           match_idx = torch.argmin(torch.abs(virtual[:,0] - jt))
           align_x[i,j] = torch.abs(jx - virtual[match_idx,1])
        
           # get first bit of data
           t2  = tracklets[j][:reg_keep,0].unsqueeze(1)
           y2  = tracklets[j][:reg_keep,1:3]
           
           pred = reg.predict(t2)
           diff = np.abs(pred - y2.data.numpy())
           
           #assign
           mdy = diff[:,1].mean()
           align_y[i,j] = mdy

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

     return align_x,align_y
    
def compute_scores_overlap(tracklets,
                           intersection_other, 
                           params, 
                           iteration, 
                           align_x,
                           align_y,
                           i = None,
                           hz = 0.1):
    
    """
    Differently than the other compute_scores functions, here we consider all pairs that have a time overlap? 
    """
    t_intersection = intersection_other["t_intersection"]
    x_intersection = intersection_other["x_intersection"]
    y_intersection = intersection_other["y_intersection"]
    big_number = params["big_number"]
    mint_int = intersection_other["mint_int"]
    maxt_int = intersection_other["maxt_int"]
    
    # align_x = align_x*0 + big_number
    # align_y = align_y*0 + big_number # overwrite old scores
    
    # go through all pairs and compute mean distance over intersection
    if i is None:
        align_x = align_x*0 + big_number
        align_y = align_y*0 + big_number # overwrite old scores
        i_list = list(range(len(tracklets)))
    else:
        align_x[i,:] = big_number
        align_x[:,i] = big_number
        align_y[i,:] = big_number
        align_y[:,i] = big_number
        i_list = [i]
    
    for i in i_list:
        if i >= t_intersection.shape[0]: break
    
        #if i %10 == 0: print("On tracklet {}".format(i))
        for j in range(len(tracklets)):
            if j == i: continue
            if j >= t_intersection.shape[0]: continue
            

            if t_intersection[i,j] > 0 and t_intersection[i,j] < 10 and x_intersection[i,j] > 0 and y_intersection[i,j] > 0:
                tmin = mint_int[i,j]
                tmax = maxt_int[i,j]
                
                
                # compute mean dist over t_overlap range at X Hz
                try: eval_t = torch.arange(tmin,tmax-3*hz,step = hz)
                except RuntimeError:
                    eval_t = [tmin,tmax]
                
                i_idxs = []
                j_idxs = []
                
                i_iter = 0
                j_iter = 0
                
                for t in eval_t:
                    try:
                        while tracklets[i][i_iter,0] < t: i_iter += 1
                        i_idxs.append(i_iter)
                        
                        while tracklets[j][j_iter,0] < t: j_iter += 1
                        j_idxs.append(j_iter)
                    except IndexError:
                        break
                
                # ensure there were some selected data points
                l = min(len(j_idxs),len(i_idxs))
                if l < 2: continue
                j_idxs = j_idxs[:l]
                i_idxs = i_idxs[:l]
                    
                # only use close comparison points to compute score
                mask = torch.where(tracklets[i][i_idxs,0]- tracklets[j][j_idxs,0] < 0.1,1,0).nonzero().squeeze(1)
                ix = tracklets[i][i_idxs,1][mask]
                iy = tracklets[i][i_idxs,2][mask]
                jx = tracklets[j][j_idxs,1][mask]
                jy = tracklets[j][j_idxs,2][mask]
                align_x[i,j] = torch.sqrt((ix-jx)**2).mean()
                align_y[i,j] = torch.sqrt((iy-jy)**2).mean()
                
                # if align_x[i,j] + align_y[i,j] < params["cutoff_dists"][iteration]:
                #      print("Tracklets {} and {} mean overlap distance {:.1f}ft and {} comparisons".format(i,j,align_x[i,j] + align_y[i,j],len(i_idxs)))
           
    align_x = torch.nan_to_num(align_x,nan = big_number)
    align_y = torch.nan_to_num(align_y,nan = big_number)

    return align_x,align_y

def compute_scores_overlap2(tracklets,
                           raster_pos,
                           params, 
                           align_x,
                           align_y,
                           i = None):    
    """
    Differently than the other compute_scores functions, here we consider all pairs that have a time overlap? 
    """
    # t_intersection = intersection_other["t_intersection"]
    # x_intersection = intersection_other["x_intersection"]
    # y_intersection = intersection_other["y_intersection"]
    big_number = params["big_number"]
    # mint_int = intersection_other["mint_int"]
    # maxt_int = intersection_other["maxt_int"]
    
    # align_x = align_x*0 + big_number
    # align_y = align_y*0 + big_number # overwrite old scores
    
    # go through all pairs and compute mean distance over intersection
    if i is None:
        align_x = align_x*0 + big_number
        align_y = align_y*0 + big_number # overwrite old scores
        i_list = list(range(len(tracklets)))
    else:
        align_x[i,:] = big_number
        align_x[:,i] = big_number
        align_y[i,:] = big_number
        align_y[:,i] = big_number
        i_list = [i]
    
    for i in i_list:
        # if i % 10 == 0:
        #     print("Computing raster-based scores for tracklet {}".format(i))
        
        
        # compare rasterized positions for i to each other object and get mean pf non-nan
        tpos = raster_pos[i] # nsamples,2
        tpos = tpos.unsqueeze(0).expand(raster_pos.shape)
        
        diff = torch.abs(raster_pos - tpos)
        mean_diff = torch.nanmean(diff,dim = 1) # n_tracklets,2
        mean_diff[i] = big_number # so that self is not a match
        
        # deal with case where there were no matches so all values are nan
        mean_diff = torch.nan_to_num(mean_diff,nan = big_number)
        
        align_x[i,:] = mean_diff[:,0]
        align_y[i,:] = mean_diff[:,1]


    return align_x,align_y    
    
def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

    
def get_msf_points(tracklet,msf_dict,extra_t = 10,freq = 0.2,direction = -1):
    """
    Returns virtual tracklet points using msf for extra_t seconds after end of tracklet, sampled at freq intervals
    
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
        if t_bin >= msf.shape[1] or t_bin < 0: break
        
        x_bin = int((x-msf_dict["start_x"]) // msf_dict["x_grid"])
        if x_bin >= msf.shape[0] or x_bin < 0: break 
        
        speed = msf[x_bin,t_bin]
        x = x + (direction * speed*freq)
        t = t + freq
        virtual.append([t,x])
    
    virtual = torch.tensor(virtual)
    return virtual

def get_msf_points_batched(tracklets,msf_dict,extra_t=10,freq=0.2,direction = -1):
    """
    Returns virtual tracklet points for all tracklets using msf for extra_t seconds after end of tracklet, sampled at freq intervals
    
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
    
    # n_tracklets 
    start_x = torch.tensor([t[-1,1] for t in tracklets])
    start_t = torch.tensor([t[-1,0] for t in tracklets])
    start_y = torch.tensor([t[-1,2] for t in tracklets])
    
    n_tracklets = len(tracklets)
    n_samples =int(np.ceil(extra_t/freq))
    virtual = torch.zeros([n_tracklets,n_samples,2])
    virtual[:,0,0] = start_t
    virtual[:,0,1] = start_x
    
    # get all appropriate lanes
    lanes = (start_y// -12).int()
    
    t_pluses = np.arange(0,extra_t,step = freq)
    for idx in range(1,len(t_pluses)):
        t_plus = t_pluses[idx]
        
        # t_plus is the additional time relative to the start_time
        
        
        # get msf t_bins
        t_bin = (((start_t+t_plus)-msf_dict["start_t"]) // msf_dict["t_grid"]).int() # it is possible that these bins extend past the edge of the MSF 
        t_bin = torch.clamp(t_bin,min = 0,max = msf.shape[2]-1)
        
        # get msf x_bins
        x_bin = ((virtual[:,idx-1,1] - msf_dict["start_x"]) // msf_dict["x_grid"]).int()
        x_bin = torch.clamp(x_bin,min = 0,max = msf.shape[1]-1)
        
        # get speeds - a grotesque triple index awaits!
        speeds = msf[lanes,x_bin,t_bin]   # this may be wrong so be aware
        
        # get x pos
        virtual[:,idx,0] = virtual[:,0,0] + t_plus
        virtual[:,idx,1] = virtual[:,idx-1,1] + (direction*speeds*freq)
        
       
    return virtual


def generate_msf(tracklets,start_time,end_time,start_x,end_x,grid_t = 2,grid_x = 50,kwidth = 7,lane = 1, SHOW = False,step = 5):
    
    # create holder matrix
    t_range = torch.arange(start_time,end_time,grid_t)
    t_range2 = torch.cat((t_range[1:],torch.tensor(end_time).unsqueeze(0)))
    x_range = torch.arange(start_x,end_x,grid_x)
    x_range2 = torch.cat((x_range[1:],torch.tensor(end_x).unsqueeze(0)))

    distance = torch.zeros(x_range.shape[0],t_range.shape[0],10)
    time     = torch.zeros(x_range.shape[0],t_range.shape[0],10) 
    
    for tdx,tracklet in enumerate(tracklets):
        
        #skip every other tracklet
        if tdx % 2 == 0:
            continue
        
        #if tdx%100 == 0: print("Adding tracklet {} to mean speed field".format(tdx))
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
        
        for i in np.arange(0,p,step = 8):
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
    for lane in range(0,10):
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

def generate_msf_failed(tracklets,start_time,end_time,start_x,end_x,grid_t = 2,grid_x = 50,kwidth = 7,lane = 1, SHOW = False,step = 5):
    direction = -1
    # create holder matrix
    t_range = torch.arange(start_time,end_time,grid_t)
    t_range2 = torch.cat((t_range[1:],torch.tensor(end_time).unsqueeze(0)))
    x_range = torch.arange(start_x,end_x,grid_x)
    x_range2 = torch.cat((x_range[1:],torch.tensor(end_x).unsqueeze(0)))

    y_range = torch.arange(0,120,12)
    y_range2 = torch.arange(12,121,12)
    
    distance = torch.zeros(x_range.shape[0],t_range.shape[0],y_range.shape[0])
    time     = torch.zeros(x_range.shape[0],t_range.shape[0],y_range.shape[0]) 
    for tdx,tracklet in enumerate(tracklets):
        
        # skip every other tracklet
        if tdx % 2 == 0:
            continue
        
        #if tdx%100 == 0: print("Adding tracklet {} to mean speed field".format(tdx))
        # get the points that belong in each time and space bin
        # yep, you guessed it, it's gonna be another 3-4D tensor op
        # there are p points. we need to determine whether each point is > t1 and < t2
        p  = tracklet.shape[0]
        tra_x = tracklet[:,1].unsqueeze(1).expand(p,x_range.shape[0])
        tra_t = tracklet[:,0].unsqueeze(1).expand(p,t_range.shape[0])
        tra_y = tracklet[:,2].unsqueeze(1).expand(p,y_range.shape[0]) * direction
        
        x1 = x_range.unsqueeze(0).expand(p,x_range.shape[0])
        x2 = x_range2.unsqueeze(0).expand(p,x_range.shape[0])
        xbin = torch.where(torch.logical_and(tra_x > x1,tra_x < x2),1,0).nonzero()
        
        t1 = t_range.unsqueeze(0).expand(p,t_range.shape[0])
        t2 = t_range2.unsqueeze(0).expand(p,t_range.shape[0])
        tbin = torch.where(torch.logical_and(tra_t > t1,tra_t < t2),1,0).nonzero()
        
        y1 = y_range.unsqueeze(0).expand(p,y_range.shape[0])
        y2 = y_range2.unsqueeze(0).expand(p,y_range.shape[0])
        ybin = torch.where(torch.logical_and(tra_y > y1,tra_y < y2),1,0).nonzero()
        
        # theres a chance that xbin falls outside of x range but the other two do not, so we have to combine into the set of indices where x,y,t all fall within the acceptable range
        combined = torch.zeros([p,4],dtype = int)
        combined[xbin[:,0],0] = xbin[:,1]
        combined[xbin[:,0],3] += 1
        combined[ybin[:,0],2] = ybin[:,1]
        combined[ybin[:,0],3] += 1
        combined[tbin[:,0],1] = tbin[:,1]
        combined[tbin[:,0],3] += 1
        keep = torch.where(combined[:,3] == 3,1,0).nonzero().squeeze(1)
        
        combined = combined[keep,:3]
        combined = combined[:-1]
        
        # note that this is a tad bit imprecise because we assign delta (t to t+1) to time t but we clamp the maximum t so that it doesn't fall of the end
        dx = tracklet[1:,1] - tracklet[:-1,1]
        dt = tracklet[1:,0] - tracklet[:-1,0]
        keep = torch.clamp(keep,0,p-2)
        dx = dx[keep]#[:min(dx.shape[0],combined.shape[0])]
        dt = dt[keep]#[:min(dt.shape[0],combined.shape[0])]
        
        
        # if necessary for mem allocation we can do this with a loop instead of a broadcast
        for idx in range(combined.shape[0]):
            
            distance[combined[idx,0],combined[idx,1],combined[idx,2]] += dx[idx]
            time[combined[idx,0],combined[idx,1],combined[idx,2]] += dt[idx] 
        
        #minmax = {}
        # for i in np.arange(0,p,step = 3):
        #     try:
        #         m = torch.where(tbin[:,0] == i,1,0).nonzero().squeeze().item()
        #         n = torch.where(xbin[:,0] == i,1,0).nonzero().squeeze().item()
        #         lane  = int((tracklet[i,2] // -12).item())
        #     except RuntimeError: continue # occurs when the point falls outside of all bin ranges
            
        #     if "{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane) not in minmax.keys():
        #         minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)] = {"xmin":tracklet[i,1],
        #                                                    "xmax":tracklet[i,1],
        #                                                    "tmin":tracklet[i,0],
        #                                                    "tmax":tracklet[i,0]}
        #     else:
        #         minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["xmin"] = min(tracklet[i,1],minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["xmin"])        
        #         minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["xmax"] = max(tracklet[i,1],minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["xmax"])   
        #         minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["tmin"] = min(tracklet[i,0],minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["tmin"])        
        #         minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["tmax"] = max(tracklet[i,0],minmax["{}:{}:{}".format(xbin[n,1].item(),tbin[m,1].item(),lane)]["tmax"])   
        
        # for key in minmax.keys():
        #     xidx = int(key.split(":")[0])
        #     tidx = int(key.split(":")[1])
        #     lidx = min(6,int(key.split(":")[2])) # make sure no out-of-bounds data
            
        #     t_elapsed = minmax[key]["tmax"] - minmax[key]["tmin"]
        #     if t_elapsed > 0.2:
        #         x_elapsed = minmax[key]["xmax"] - minmax[key]["xmin"]
            
        #         distance[xidx,tidx,lidx] += x_elapsed
        #         time[xidx,tidx,lidx] += t_elapsed
    
    all_msf_unsmooth = []    
    all_msf= []
    
    SHOW = False
    
    for lane in range(0,10):
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


        
def track_chunk(data_dir,tree,q = None,SHOW = False):
   tm = Timer()
   msf_dict = None 
   
   
   
   tm.split("data loading")
   
   # parse params
   params = tree["pass_params"]
   start_x    = tree["start_x"]
   end_x      = tree["end_x"]
   start_time = tree["start_time"]
   end_time   = tree["end_time"]
   data_dir = params["data_dir"]  
   path = "{}/tracklets_{}_{}_{}_{}.cpkl".format(data_dir,start_time,end_time,start_x,end_x)
   
   t_width = end_time - start_time

   
   tree2 = copy.deepcopy(tree)
   interface_x = tree2["interface_x"]
   interface_t = tree2["interface_t"]
   interface_width_t = tree2["interface_width_t"]
   interface_width_x = tree2["interface_width_x"]
   
   
   # if get_tracklets is passed interface parameters, it will subdivide tracklets into 3 catergories (with 2 seam idxs instead of 1)
   # first half, second half , irrelevant (pass through to end of process as tracklets_complete)
   
   tracklets,seams,passthrough = get_tracklets(data_dir,tree)
   
   if len(tracklets) == 0: raise Exception
   
   seam_idx = None
   if len(seams) == 2:
       seam_idx = seams[0]
       print("\nUSING SEAM IDX {}\n".format(seam_idx)) 
       if q is not None: q.put([tree,"\nUSING SEAM IDX {}\n".format(seam_idx)])
   
   # if interface parameters were specified, modify the start and end params to reflect the interface
   if interface_x is not None:
       start_x = max(start_x,interface_x - interface_width_x)
       end_x   = min(end_x,interface_x + interface_width_x)
       if q is not None: q.put([tree,"Using interface x width of {}, with {} tracklets and {} pass_through tracklets".format(interface_width_x,len(tracklets),len(passthrough))])

   elif interface_t is not None:
       start_time = max(start_time,interface_t - interface_width_t)
       end_time = min(end_time,interface_t + interface_width_t)
       if q is not None: q.put([tree,"Using interface t width of {}, with {} tracklets and {} pass_through tracklets".format(interface_width_t,len(tracklets),len(passthrough))])
   
   tm.split("queue write")
   
   tracklets_complete = []
   
   
   #if q is not None:        q.put([tree,tracklets,tracklets_complete,-1,None])
   
    
   for iteration in range(len(params["t_thresholds"])):
       start_count = len(tracklets)
       iteration_start_time = time.time()
       
       # compute interesections
       t_threshold = params["t_thresholds"][iteration]
       x_threshold = params["x_thresholds"][iteration]
       y_threshold = params["y_thresholds"][iteration]
       min_regression_length = params["min_regression_lengths"][iteration]
       reg_keep = params["reg_keeps"][iteration]
       cutoff_dist = params["cutoff_dists"][iteration]
       big_number = params["big_number"]
       min_duration = params["min_duration"]
       x_margin = params["x_margin"]
       t_margin = params["t_margin"]
       mode = params["mode"][iteration]

       tm.split("intersections")
       #if q is not None: q.put("ITERATION {}: Calculating initial intersections for {} {} {} {}".format(iteration,start_x,end_x,start_time,end_time))

       intersection,intersection_other = compute_intersections(tracklets,t_threshold, x_threshold, y_threshold,seam_idx=seam_idx)

       # compute score via one of 3 methods - linear, msf virtual trajectory, overlap average distance
       try: 
           align_x
       except:
           align_x = torch.zeros(intersection.shape)  + params["big_number"]
           align_y = torch.zeros(intersection.shape)  + params["big_number"]
           align_xb = torch.zeros(intersection.shape) + params["big_number"]
           align_yb = torch.zeros(intersection.shape) + params["big_number"]
       
       #if q is not None: q.put("ITERATION {}: Calculating initial {} scores for {} {} {} {}".format(iteration,mode,start_x,end_x,start_time,end_time))

       if mode == "linear":
           tm.split("scores_linear_init")
           align_x,align_y,align_xb,align_yb = compute_scores_linear(tracklets,intersection,params,iteration,align_x,align_y,align_xb,align_yb)
       if mode == "msf": # overwrite align_x with msf-based distances
           if msf_dict is None or params["recompute_msf"][iteration]:
               tm.split("get_msf")
               msf_dict = generate_msf(tracklets, start_time, end_time, start_x, end_x)
               #if q is not None:   q.put("Got MSF for  {} {} {} {}".format(start_x,end_x,start_time,end_time))
               
           align_x = torch.zeros(intersection.shape)  + params["big_number"]
           align_y = torch.zeros(intersection.shape)  + params["big_number"]
           tm.split("scores_msf_init")
           align_x,align_y = compute_scores_msf2(tracklets,intersection,params,iteration,align_x,align_y, msf_dict)
       elif mode == "overlap":
           try: raster_pos
           except: 
               tm.split("get_raster_pos")
               raster_pos = compute_raster_pos(tracklets, start_time, end_time, start_x, end_x)
           tm.split("scores_overlap_init")
           align_x,align_y = compute_scores_overlap2(tracklets,raster_pos,params,align_x,align_y)



       #if q is not None:   q.put("Finished initial {} scores for {} {} {} {}".format(mode,start_x,end_x,start_time,end_time))
 

       FINAL = False
       removals = []
       CHANGEME = 0
       
       tm.split("main_check_for_termination")
       for i in range(len(tracklets)):
            if torch.min(tracklets[i][:,1]) < start_x + x_margin or torch.max(tracklets[i][:,0]) > end_time - t_margin:
                if torch.max(tracklets[i][:,1]) > end_x  - x_margin or  torch.min(tracklets[i][:,0]) < start_time + t_margin:
                    tracklets_complete.append(tracklets[i])
                    CHANGEME += 1
                    removals.append(i)
                    #print("BEGINNING - Tracklet {} added to finished queue".format(i))
                    #if q is not None:  q.put("BEGINNING - Tracklet {} added to finished queue".format(i))
       
       while len(tracklets) > 0 :
               i = None

               tm.split("main_mask_long")
           
               if (len(tracklets) - CHANGEME) % 10 == 0:
                   durations = [t[-1,0] - t[0,0] for t in tracklets]
                   durations_complete = [t[-1,0] - t[0,0] for t in tracklets_complete] + [1]
                   mean_duration = sum(durations) / len(durations)
                   mean_duration_complete = sum(durations_complete)/len(durations_complete) 
                   #print("{} Finished, {} Unfinished tracklets with mean duration: {:.1f}s,{:.1f}s".format(len(tracklets_complete),len(tracklets)-CHANGEME,mean_duration,mean_duration_complete))
                   #if q is not None:  q.put("{} Finished, {} Unfinished tracklets with mean duration: {:.1f}s,{:.1f}s".format(len(tracklets_complete),len(tracklets)-CHANGEME,mean_duration,mean_duration_complete))

               ## A disgusting masking operation that should make you queasy
               # update alignment matrices according to tracklet length logic
               # if tracklet i is too short, but tracklet j is not, replace tracklet i score with tracklet j backwards score   align_x[i,j] <- align_xb[i,j]
               # if tracklet j is too short but tracklet i is not, replace tracklet j backwards score with tracklet i score    align_xb[i,j] <- align_x[i,j]   
               
               if False and mode == "linear":
                    tracklet_lengths = torch.tensor([t.shape[0] for t in tracklets]).unsqueeze(1).expand(len(tracklets),len(tracklets))
                    mask_ilong = torch.where(tracklet_lengths > min_regression_length,1,0) # 1 where element i is long, 0 otherwise
                    mask_jlong = torch.where(tracklet_lengths.transpose(1,0) > min_regression_length,1,0) # 1 where element j is long, 0 otherwise
            
                    align_x = align_x*mask_ilong + (1-mask_ilong)*(mask_jlong*align_xb + (1-mask_jlong)*big_number)
                    align_y = align_y*mask_ilong + (1-mask_ilong)*(mask_jlong*align_yb + (1-mask_jlong)*big_number)
                    align_xb = align_xb*mask_jlong + (1-mask_jlong)*(mask_ilong*align_x + (1-mask_ilong)*big_number)
                    align_yb = align_yb*mask_jlong + (1-mask_jlong)*(mask_ilong*align_y + (1-mask_ilong)*big_number)
        
               tm.split("main_get_best")
 
               # compute aggregate scores as a combo of above elements
               if mode == "linear":
                   scores = (torch.sqrt(align_x**2 + align_y**2)  + torch.sqrt(align_xb**2 + align_yb**2)) /2
               elif mode == "msf":
                   scores = torch.sqrt(align_x**2 + align_y**2)
               elif mode == "overlap":
                   scores = torch.sqrt(align_x**2 + align_y**2)
            
               ### select the best-scoring association - flatten these two tracklets into a single tracklet
               # i = merged idx
               # j = deleted idx
               best_idx = torch.argmin(scores)
               best_score = torch.min(scores)
               #keep = [_ for _ in range(len(tracklets))]
               I_COMPLETE = False
               
               if best_score > cutoff_dist: 
                   tm.split("main_remove_short")

                   #print("WARNING: best score {:.1f} > cutoff distance {}. Consider Terminating".format(best_score,cutoff_dist))
                   if not FINAL:
                        FINAL = True
                        #if q is not None: q.put("Final Iteration")
                        
                        if iteration == 4:
                            # remove all tracklets that are more or less entirely subcontained within another tracklet
                             #durations = (intersection_other["max_t"].transpose(1,0) - intersection_other["min_t"] - t_threshold)[:,0]
                            
                            durations = torch.tensor([t[-1,0] - t[0,0] for t in tracklets])
                            
                            #keep = torch.where(durations > min_duration,1,0).nonzero().squeeze(1)
                            #tracklets = [tracklets[k] for k in keep]
                            
                            removals += torch.where(durations > min_duration,0,1).nonzero().squeeze(1).tolist()
                            
                            
               if not FINAL:
                   tm.split("main_merge")

                   
                   i,j = [(best_idx // len(tracklets)).item() , (best_idx % len(tracklets)).item()]
                   
                   
                   # check whether the merged tracklet would have too much jitter
                   temp_joined_tracklet =  torch.cat((tracklets[i],tracklets[j]),dim = 0)
                   temp_joined_tracklet  = temp_joined_tracklet[temp_joined_tracklet[:,0].sort()[1]] 
                   
                   # compute variation proxy
                   diff = torch.abs(temp_joined_tracklet[1:,1] - temp_joined_tracklet[:-1,1])
                   diff = diff.sort()[0]
                    
                   # if there are more than X jumps of size Y, do not allow this match
                   max_jump = 20
                   max_jump_count = 3
                   if diff[-min(max_jump_count,len(diff)-1)] > max_jump:
                        align_x[i,j] = big_number
                        align_y[i,j] = big_number
                        align_xb[i,j] = big_number
                        align_yb[i,j] = big_number
                        #if q is not None: q.put("-------------- Rejected match between tracklets {} and {} because jumpiness exceeded {:.1f}ft".format(i,j,diff[-min(max_jump_count,len(diff)-1)] ))
                        continue # this join is not made
                   
                   SHOW2 = False
                   if SHOW2: 
                       color2 = "b"
                       if FINAL: color2 = "r"
                
                       plt.scatter(tracklets[i][:,0],tracklets[i][:,1], c = "g")
                       plt.scatter(tracklets[j][:,0],tracklets[j][:,1], c = color2)
                       if mode == "linear":
                           plt.title(          "X:{:.1f}ft, Y:{:.1f}ft, {:.2f} variation ratio ".format((align_x[i,j]+align_xb[i,j])/2   ,(align_y[i,j]+align_yb[i,j])/2    ,max_jump))
                      
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
                       
                       else:
                            plt.title(          "X:{:.1f}ft, Y:{:.1f}ft, {} variation ratio".format(align_x[i,j], align_y[i,j], max_jump))
                       
                       plt.show()
                   
                   SHOW3 = False
                   lane = 1
                   if SHOW3 and CHANGEME % 10 == 0: 
                       plt.figure(figsize = (20,40))
                       for tidx in np.arange(0,len(tracklets),step = 1):
                               t = tracklets[tidx]
                                 
                               keep = torch.where(torch.logical_and(t[:,2] < -12*lane,t[:,2] > -12*(lane+1)),1,0).nonzero().squeeze(1)
                               if len(keep) == 0: continue 
                           
                               c = colors[tidx]
                               
                               tt = t[keep,:]
                               plt.plot(tt[:,0],tt[:,1],c=c,linewidth=2)
                               
                               if msf_dict is not None:
                                   virtual = get_msf_points(t,msf_dict,extra_t = pass_params["t_thresholds"][-2])
                                   plt.plot(virtual[:,0],virtual[:,1],":",c=(0.2,0.2,0.2))
                           
                       for tidx in np.arange(0,len(tracklets_complete),step = 1):
                                 t = tracklets_complete[tidx]
                                 keep = torch.where(torch.logical_and(t[:,2] < -12*lane,t[:,2] > -12*(lane+1)),1,0).nonzero().squeeze(1)
                                 if len(keep) == 0: continue 
                         
                                 
                                 c = (0,0,1)
                                 plt.plot(t[keep,0],t[keep,1],c=c,linewidth=2)    
                                 #plt.plot(t[:,0],t[:,1],c = (1,1,1),linewidth=2)
                       plt.title("Iteration {}, {} / {} tracklets".format(iteration,len(tracklets),len(tracklets_complete)))
                       plt.show()
                       
                   if i >j: # ensure j > i for deletion purposes
                       j1 = j
                       j = i
                       i = j1
                       
                  
                   
                    
                   # combine two tracklet arrays
                   tracklets[i] = temp_joined_tracklet
                   
                   # sort by time  - already done for temp_joined_tracklet
                   # tracklets[i]  = tracklets[i][tracklets[i][:,0].sort()[1]] 
                   
                   # remove larger idx from tracklets
                   #del tracklets[j]
                   
                   ## check whether the tracklet has a valid source and sink - if so remove it from further calculation
                   tm.split("main_check_for_termination")
                    # if torch.min(tracklets[i][:,1]) < start_x + x_margin or torch.max(tracklets[i][:,0]) > end_time - t_margin:
                    #     if torch.max(tracklets[i][:,1]) > end_x  - x_margin or  torch.min(tracklets[i][:,0]) < start_time + t_margin:
                    #         tracklets_complete.append(tracklets[i])
                    #         I_COMPLETE = True
                    #        #print("Tracklet {} added to finished queue".format(i))
                    #        #if q is not None:  q.put("Tracklet {} added to finished queue".format(i))
                           
                   SOURCE = False
                   for zone in zones["WB"]["source"].values():
                       if tracklets[i][0,1] > zone[0] and tracklets[i][0,1] < zone[1] and tracklets[i][0,2] > zone[2] and tracklets[i][0,2] < zone[3]:
                           SOURCE = True
                           break
                   if not SOURCE and torch.max(tracklets[i][:,1]) > tree["end_x"]  - x_margin or  torch.min(tracklets[i][:,0]) < tree["start_time"] + t_margin: # chunk boundaries are the source
                       SOURCE = True
                        
                   if SOURCE:
                       SINK = False
                       for zone in zones["WB"]["sink"].values():
                           if tracklets[i][-1,1] > zone[0] and tracklets[i][-1,1] < zone[1] and tracklets[i][-1,2] > zone[2] and tracklets[i][-1,2] < zone[3]:
                               SINK = True
                               break
                       if not SINK and torch.min(tracklets[i][:,1]) < tree["start_x"] + x_margin or torch.max(tracklets[i][:,0]) > tree["end_time"] - t_margin: # chunk boundaries are the sink
                           SINK = True
                            
                   if SOURCE and SINK:
                       tracklets_complete.append(tracklets[i])
                       I_COMPLETE = True
                           
                   # update all score matrices t_intersection, x_intersection, y_intersection, intersection, x_align, y_align
                   # if tracklet i was removed, all we need to do is remove row and column i and j
                   # otherwise, we need to remove column and row j, and update column and row i
                   
                   #keep.remove(j)
                   if j in removals:
                       if q is not None: 
                           q.put([tree,"WARNING, adding a duplicate to the removal queue"])
                       else:
                           print("WARNING - added a duplicate to the removal queue")
                   
                       
                   removals.append(j)
                   
                   CHANGEME += 1
                   
                   if I_COMPLETE:
                        #keep.remove(i) 
                        removals.append(i)

               tm.split("intersect_update")
               if not I_COMPLETE and not FINAL: # if I_COMPLETE, row and column i will be zeroed out at the bottom so we don't need to do anything
                   # in the other case, we simply assume that intersection [i+j] will include all the intersections of i and of j
                   #try: 
                       intersection,intersection_other = compute_intersections(tracklets, t_threshold, x_threshold, y_threshold,i = i,intersection = intersection,intersection_other = intersection_other,j = j)                   
                   #except UnboundLocalError: 
                   #    pass # this happen
                   
                
               # we are going to change this block so that it is only run every 50 or so joins
               if (CHANGEME+1) >31  or FINAL:
                   #q.put("We got here")
                   
                   tm.split("main_delete")

                   removals = list(set(removals))  # a tracklet can be removed due to duration criteria and due to merging
                   removals.sort()
                   removals.reverse()
                   
                   # seam_idx only changes when tracklets before seam_idx are removed - count these and decrement seam_idx accordingly
                   if seam_idx is not None:
                       decrement = 0
                       for removal in removals:
                           if removal < seam_idx:
                               decrement += 1
                       seam_idx -= decrement

                   keep = [_ for _ in range(len(tracklets))]
                   for r in removals:
                       keep.remove(r)

                   keep = torch.tensor(keep,dtype = int)
                       
                   # delete portions of matrices 
                   intersection   = intersection[keep,:][:,keep]
                   align_x = align_x[keep,:][:,keep]
                   align_y = align_y[keep,:][:,keep]
                   align_xb = align_xb[keep,:][:,keep]
                   align_yb = align_yb[keep,:][:,keep]
                   if mode == "overlap":
                       raster_pos = raster_pos[keep,:,:]

                   # TODO - we don't need this if we're doing the intersection combination trick 
                   # for mat in intersection_other:
                   #      intersection_other[mat] = intersection_other[mat][keep,:][:,keep]
                    
                   tracklets = [tracklets[k] for k in keep]
                   # for removal in removals:
                   #     del tracklets[removal]
                   
                   removals = []
                   CHANGEME = 0

               # tm.split("intersect_update")
               # intersection,intersection_other = compute_intersections(tracklets, t_threshold, x_threshold, y_threshold,i = i,intersection = intersection,intersection_other = intersection_other,seam_idx = seam_idx)  
               
               tm.split("temp_pseudo_delete")
               # so that no removed objects can be matched
               intersection[removals,:] = 0
               intersection[:,removals] = 0
               # intersection_other['t_intersection'][removals,:] = 0
               # intersection_other['t_intersection'][:,removals] = 0
            
               # update scores
               if i is not None and i <= len(tracklets) -1 and not I_COMPLETE:# if i and j were both removed, no need to update any scores 
                     if params["mode"][iteration] == "linear":
                         tm.split("scores_linear_update")
                         align_x,align_y,align_xb,align_yb = compute_scores_linear(tracklets,intersection,params,iteration,align_x,align_y,align_xb,align_yb,i = i)
                     elif params["mode"][iteration] == "msf":
                         tm.split("scores_msf_update")
                         align_x,align_y = compute_scores_msf2(tracklets,intersection,params,iteration,align_x,align_y,msf_dict,i = i)
                     elif params["mode"][iteration] == "overlap":
                         tm.split("scores_overlap_update")
                         #print(raster_pos.shape,intersection.shape,align_x.shape,len(tracklets),FINAL)
                         raster_pos = compute_raster_pos(tracklets, start_time, end_time, start_x, end_x,i=i,raster_pos = raster_pos)
                         #print(raster_pos.shape,intersection.shape,align_x.shape,len(tracklets))
                         align_x,align_y = compute_scores_overlap2(tracklets,raster_pos,params,align_x,align_y,i=i)
               
               align_x[removals,:] = big_number
               align_x[:,removals] = big_number  

               if FINAL: break
           
       # end of iteration
       # resample
       if False and iteration == 8 and t_width <40 and t_width > 20:
           tm.split("resample")
           tracklets,tracklets_complete = resample(tracklets,tracklets_complete)
       
       tm.split("cleanup")
       assert len(removals) == 0,  "Removals is non-empty at end of iteration."
       
       if q is not None: q.put([tree,"ITERATION {}: {} -> {} tracklets, {:.1f}s elapsed".format(iteration,start_count,len(tracklets),time.time() - iteration_start_time)])
       
       if SHOW:
            lane = 1
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
         
       tm.split("queue write")     
       if q is not None:
           if iteration < len(params["t_thresholds"]) -1:
               q.put([tree,[t.data.numpy() for t in tracklets+passthrough],[t.data.numpy() for t in tracklets_complete],iteration,None])
           # else:
           #     q.put([tracklets,tracklets_complete,iteration,None])
   # save results 
   tm.split("data write")
   
   with open(path,"wb") as f:
       pickle.dump([tracklets+passthrough,tracklets_complete],f)
   tm.split("cleanup")
   
   
   
   if q is not None: q.put([tree,[t.data.numpy() for t in tracklets+passthrough],[t.data.numpy() for t in tracklets_complete],iteration,tm.bins()])
   if q is not None: q.put([tree,"DONE"]) 
    
   #time.sleep(1) # hoping to flush queue?
   return #tracklets,tracklets_complete

def get_tracklets(data_dir,tree,direction = -1,SHOW = False):
    """ """
    seams = []
    tracklets_complete = []
    
    interface_x = tree["interface_x"]
    interface_t = tree["interface_t"]
    interface_width_t = tree["interface_width_t"]
    interface_width_x = tree["interface_width_x"]
    
    try: 
        tracklets = []
        
        if len(tree["dep"]) == 0: raise FileNotFoundError  # this is bad and should be changed
        
        for path in tree["dep"]:
        
            with open(path,"rb") as f:
                d,d_term = pickle.load(f)
            
            if interface_x is None and interface_t is None:
                tracklets += d
                tracklets += d_term
                
            elif interface_x is not None:
                dd = d + d_term
                # here we need to parse tracklets to keep only those that fall within the interface boundary
                for item in dd:
                    if (item[0,1] < interface_x + interface_width_x and item[0,1] > interface_x - interface_width_x) or (item[-1,1] < interface_x + interface_width_x and item[-1,1] > interface_x - interface_width_x):
                        tracklets.append(item)
                    else:
                        tracklets_complete.append(item)
            
            elif interface_t is not None:
                dd = d + d_term
                # here we need to parse tracklets to keep only those that fall within the interface boundary
                for item in dd:
                    if (item[0,0] < interface_t + interface_width_t and item[0,0] > interface_t - interface_width_t) or (item[-1,0] < interface_t + interface_width_t and item[-1,0] > interface_t - interface_width_t):
                        tracklets.append(item)
                    else:
                        tracklets_complete.append(item)
            
            seams.append(len(tracklets)) # note that this seam scheme relies on the text-on-page ordering of tracklet data left to right, up to down perhaps?
                    
                    
                
        print("Loaded cached tracklet blocks, {} total".format(len(tracklets)))
        
        
        
        
    except:
        for file in os.listdir(data_dir):
            if "clusters" in file:
                pieces = file.split(".")[0].split("_")
                if int(pieces[1]) <= tree["start_time"] and int(pieces[2]) >= tree["end_time"] and int(pieces[3]) <= tree["start_x"] and int(pieces[4]) >= tree["end_x"]:
                    path = os.path.join(data_dir,file)
                        #path = "{}/clusters_{}_{}_{}_{}.npy".format(data_dir,tree["start_time"],tree["end_time"],tree["start_x"],tree["end_x"])
                    det = torch.from_numpy(np.load(path))
                    time_idxs = torch.where(torch.logical_and(det[:,0] > tree["start_time"],det[:,0] < tree["end_time"]),1,0)
                    space_idxs = torch.where(torch.logical_and(det[:,1] > tree["start_x"],det[:,1]<tree["end_x"]),1,0)
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
        lane = 1
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
    
    return tracklets,seams,tracklets_complete


def plot_tree(tree,lane = 1,boxes = True, trac = 20, COLOR = False, msf_dict = None):
    """
    If node has plottable material, plot it, otherwise go to each child node and plot
    """
    durations = [1]
    if tree["data"] is not None:
        tracklets,tracklets_complete,iterc = tree["data"]
        #title = str(iteration) + ","
        
        if boxes:
            w = tree["end_time"] - tree["start_time"]
            h = tree["end_x"] - tree["start_x"]
            Rectangle((tree["start_time"],tree["start_x"]),w,h,color = (0,0,0),linewidth = 4)
            
            xx = tree["start_time"],tree["end_time"],tree["end_time"],tree["start_time"],tree["start_time"]
            yy = tree["end_x"],tree["end_x"],tree["start_x"],tree["start_x"],tree["end_x"]
            if iterc != "DONE":
                plt.plot(xx,yy,c = (0,0,0),linewidth = 5)
                
                text = "it.{},{}".format(iterc,len(tracklets))
                plt.text(tree["start_time"] + w/2, tree["start_x"]+h/2, text,fontsize = 20)
                
            else:
                plt.plot(xx,yy,":",c = (0.6,0.6,0.6),linewidth = 3)
                
           
        if trac > 0:        
            # plot incomplete tracklets
            for tidx in np.arange(0,len(tracklets),step = trac):
                    t = tracklets[tidx]
                    durations.append(t[-1,0]-t[0,0])
                      
                    keep = torch.where(torch.logical_and(t[:,2] < -12*lane,t[:,2] > -12*(lane+1)),1,0).nonzero().squeeze(1)
                    if len(keep) == 0: continue 
                
                    if (iterc == "DONE" or iterc == len(pass_params["cutoff_dists"])-1) and not COLOR:
                        c = (0.6,0.6,0.6)
                    else:
                        c = colors[tidx]
                    
                    tt = t[keep,:]
                    plt.plot(tt[:,0],tt[:,1],c=c,linewidth=2)
                    
                    if msf_dict is not None:
                        virtual = get_msf_points(t,msf_dict,extra_t = pass_params["t_thresholds"][-2])
                        plt.plot(virtual[:,0],virtual[:,1],":",c=(0.2,0.2,0.2))
                
            for tidx in np.arange(0,len(tracklets_complete),step = trac):
                      t = tracklets_complete[tidx]
                      durations.append(t[-1,0]-t[0,0])
                      keep = torch.where(torch.logical_and(t[:,2] < -12*lane,t[:,2] > -12*(lane+1)),1,0).nonzero().squeeze(1)
                      if len(keep) == 0: continue 
              
                      if iterc == "DONE":
                          c = (0,0,1)
                      else:
                          c = (0,0,1)
                      plt.plot(t[keep,0],t[keep,1],c=c,linewidth=2)    
                      #plt.plot(t[:,0],t[:,1],c = (1,1,1),linewidth=2)
    
    else:
        for child in tree["children"]:
            durations += plot_tree(child,lane = lane,boxes = boxes, trac = trac)
    
    
    return durations
    #plt.title("After iterations: {}".format(it_list))
    
    
    
def put_data_in_tree(data,tree,start_time,end_time,start_x,end_x):
    if tree["start_time"] == start_time and tree["end_time"] == end_time and tree["start_x"] == start_x and tree["end_x"] == end_x:
        tree["data"] = data
    elif len(tree["children"]) > 0:
        for child in tree["children"]:
            put_data_in_tree(data,child,start_time,end_time,start_x,end_x)
    else:
        pass #print("Failed to put data in tree: {} {} {} {}".format(start_time,end_time,start_x,end_x))

def is_covered(data_dir,job):
    for file in os.listdir(data_dir):
        if ".npy" in file: continue
        pieces = file.split(".")[0].split("_")
        if int(pieces[1]) <= job["start_time"] and int(pieces[2]) >= job["end_time"] and int(pieces[3]) <= job["start_x"] and int(pieces[4]) >= job["end_x"]:
            return True
        
    return False


def worker_wrapper(data_dir,message_q,job_q,process_lock):
    
    while True:
        
        with process_lock:
            try:
                job = job_q.get(timeout = 0)
            except queue.Empty:
                time.sleep(1)
                continue
        
        if job is not None:
            # start job
            if message_q is not None: message_q.put([job,"worker_wrapper recieved a new job and is starting"])
            track_chunk(data_dir,job,message_q)
        
        else:
           break # terminate if you receive a None which will be sent at the end of processing     
            


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass




#%% preliminaries


    # specifies parameters for a single iteration (i.e. what type, what cutoffs and thresholds)
    iter_pack = {} # list of iter_packs is an additional attribute added to each node
    
    # block_pack contains all of the parameters for a single block, including a series of iter_packs as well as parameters such as whether to use a limited width interface or seam_idxs
    block_pack = {} # these parameters are stored in the tree nodes
    
    # run pack contains a series of block_packs arranged in a tree or rolling list
    run_pack = {} #- this is just the tree

    df   = "/home/worklab/Documents/datasets/I24-V/final_detections.npy"
    data_dir = "data/1"
    
    # select a time and space chunk to work on 
    #data/0
    # start_time = 0
    # end_time = 3600
    # start_x = 7000
    # end_x  = 10000
    # direction = -1
    
    #data/1
    start_time = 0
    end_time = 900
    start_x = 7000
    end_x  = 10000
    direction = -1
    space_chunk = 3000
    time_chunk  = 150

    
    #data/3
    # start_time = 1000
    # end_time = 1720
    # start_x = 3000
    # end_x  = 8000
    # direction = -1
    # space_chunk = 2500
    # time_chunk = 45
    
    #data/4
    # start_time = 2000
    # end_time = 2060
    # start_x = 0
    # end_x  = 22000
    # direction = -1
    
    #data/bm1
    # start_time = 0
    # end_time = 600
    # start_x = 11000
    # end_x  = 14000
    # direction = -1
    # space_chunk = 3000
    # time_chunk  = 75

    #data/bm2
    # start_time = 1000
    # end_time = 1060
    # start_x = 15000
    # end_x  = 18000
    # direction = -1
    # space_chunk = 3000
    # time_chunk =75

    
    delta = 0.3
    phi = 0.4

#%% step 1
    # get tree for sub-clustering
    tree = hierarchical_tree(start_x, end_x, start_time, end_time, space_chunk, time_chunk, {}) # dummy dict is pass params, not necessary for this stage
    chunks = get_tree_leaves(tree)
    
    
    # load detections
    t1 = time.time()
    det = np.load(df)
    det = torch.from_numpy(det)
    det = det[det[:, 0].sort()[1]]
    
    print("Loading data took {:.1f}s".format(time.time() - t1))
    print("For time window [{:.1f}s,{:.1f}s], space window [{:.1f}ft,{:.1f}ft], --> {} detections".format(start_time,end_time,start_x,end_x,det.shape[0]))    

    start = time.time()
    process_list = []
    for chunk in chunks: # leave one chunk for main process
        
        # check whether this chunk is already covered
        COVERED = False
        for file in os.listdir(data_dir):
            if "clusters" in file:
                pieces = file.split(".")[0].split("_")
                if int(pieces[1]) <= chunk["start_time"] and int(pieces[2]) >= chunk["end_time"] and int(pieces[3]) <= chunk["start_x"] and int(pieces[4]) >= chunk["end_x"]:
                    print("Chunk is already covered by file {}".format(file))
                    COVERED = True
                    break
                
        if not COVERED:
            p = mp.Process(target=cluster,args = (det,chunk["start_time"],chunk["end_time"],chunk["start_x"],chunk["end_x"],direction,delta,phi,data_dir,)) 
            p.start()
            process_list.append(p)
            print("Process started: {},{},{},{}".format(chunk["start_time"],chunk["end_time"],chunk["start_x"],chunk["end_x"]))
       
    print("Started {} clustering processes".format(len(process_list)))
    
    
    
    # chunk = chunks[-1]
    # cluster(det,chunk["start_time"],chunk["end_time"],chunk["start_x"],chunk["end_x"],direction,delta = delta,phi=phi,data_dir = data_dir)
    
    print("Finished on-process clustering. Waiting for other clustering operations to terminate")
    
    try:
        for p in process_list:
            p.join()
    
    except KeyboardInterrupt:
        for p in process_list:
            p.terminate()
            p.kill()
    
        
    end = time.time()
    print("All clustering operations took {:.1f}s \n\n".format(end-start))
    
   
    
    
    
    #%% step 2 hierarchical wrapper
    space_chunk = 3000
    time_chunk = 30
    
    
    
    # assemble the pass_param stack - a series of pass params along with chunk sizes for which they are valid [[maxt_chunk,maxx_chunk,pass_params],[],..] in order from smallest to largest chunk
    pass_param_stack = []
    
    #param set 1
    nit = 18
    maxx_chunk = 3000
    maxt_chunk = 120
    pass_params = {
    "mode"                   : ["linear","linear","linear","linear","linear","msf"   ,"msf"   ,"msf"   ,"msf"   ,"msf"   ,"msf"   ,"msf"   ,"msf"   ,"msf"   ,"msf"   ,"msf"   ,"msf"   ,"overlap"],
    "t_thresholds"           : [1       ,2       ,2       ,2       ,2       ,3       ,3       ,3       ,3       ,3       ,3       ,3       ,3       ,3       ,3       ,3       ,6       ,0        ],                       # tracklets beyond this duration apart from one another are not considered
    "cutoff_dists"           : [3       ,5       ,7       ,8       ,10      ,10      ,10      ,10      ,12      ,15      ,15      ,17      ,18      ,20      ,22      ,22      ,22      ,25       ],      # tracklets with a minimum scored match of > _ ft apart are not matched
    "y_thresholds"           : [2       ,5       ,5       ,5       ,6       ,6       ,7       ,8       ,8       ,8       ,8       ,8       ,8       ,8       ,8       ,8       ,8       ,6        ],                    # tracklets beyond this distance apart from one another are not considered
    "recompute_msf"          : [False   ,False   ,False   ,False   ,False   ,True    ,False   ,False   ,False   ,False   ,False    ,False   ,False   ,False   ,False   ,False   ,False   ,False    ],  
    "reg_keeps"              : [30 for _ in range(nit)],                               # the last _ points are used to fit a linear regression and the first _ points from the next tracklet are scored
    "min_regression_lengths" : [80 for _ in range(nit)],                     # tracklets less than _ in length are not used to fit a linear regression
    "x_thresholds"           : [400 for _ in range(nit)],                              # tracklets beyond this distance apart from one another are not considered
    "min_duration"           : 1,
    "x_margin"               : 100,
    "t_margin"               : 3,
    "big_number"             : 10000,
    "data_dir"               : data_dir,
    "INTERFACE"              : False,
    }
    assert nit == len(pass_params["t_thresholds"])
    pass_param_stack.append([maxt_chunk, maxx_chunk,pass_params])

    # param set 2
    nit = 5
    maxx_chunk = 100000
    maxt_chunk = 100000
    pass_params = {
    "mode"                   : ["msf"   ,"msf"   ,"msf"   ,"msf"   ,"overlap"],
    "t_thresholds"           : [3       ,3       ,3       ,6       ,0],                       # tracklets beyond this duration apart from one another are not considered
    "cutoff_dists"           : [20      ,22      ,22      ,22      ,25],      # tracklets with a minimum scored match of > _ ft apart are not matched
    "y_thresholds"           : [8       ,8       ,8       ,8       ,6],                    # tracklets beyond this distance apart from one another are not considered
    "recompute_msf"          : [True   ,False   ,False   ,False   ,False   ],  
    "reg_keeps"              : [30 for _ in range(nit)],                               # the last _ points are used to fit a linear regression and the first _ points from the next tracklet are scored
    "min_regression_lengths" : [80 for _ in range(nit)],                     # tracklets less than _ in length are not used to fit a linear regression
    "x_thresholds"           : [400 for _ in range(nit)],                              # tracklets beyond this distance apart from one another are not considered
    "min_duration"           : 1,
    "x_margin"               : 100,
    "t_margin"               : 3,
    "big_number"             : 10000,
    "data_dir"               : data_dir,
    "INTERFACE"              : True,

    }
    assert nit == len(pass_params["t_thresholds"])
    pass_param_stack.append([maxt_chunk, maxx_chunk,pass_params])
    
    
    tree = hierarchical_tree(start_x, end_x, start_time, end_time, space_chunk, time_chunk,pass_param_stack)
    #tree = time_ordered_tree(start_x, end_x, start_time, end_time, time_chunk)
    #chunks = get_tree_leaves(tree)
    

    

   
    
    
    n_workers = 4

    # traverse the tree and for each node, create a list of files that must exist in order to start
    all_jobs = flatten_tree(copy.deepcopy(tree), data_dir)
    all_jobs.reverse()
    n_jobs = len(all_jobs)

    terminated =[]
    started = []
    
    process_queue = mp.Queue()
    process_lock = mp.Lock()
    
    removals = []
    loaded_files = []
    plot_counter = 0
    
    
    
    try:
        
    
        # load completed data for plotting
        for job in all_jobs:
            for file in os.listdir(data_dir):
                if ".npy" in file: continue
                pieces = file.split(".")[0].split("_")
                if int(pieces[1]) <= job["start_time"] and int(pieces[2]) >= job["end_time"] and int(pieces[3]) <= job["start_x"] and int(pieces[4]) >= job["end_x"]:
                    path = os.path.join(data_dir,file)
                    
                    
                    if file not in loaded_files:
                        with open(path,"rb") as f:
                            tracklets,tracklets_terminated = pickle.load(f)
                            terminated.append(job)
                            put_data_in_tree([tracklets,tracklets_terminated,"DONE"], tree, job["start_time"], job["end_time"], job["start_x"], job["end_x"])
                            print("Reloaded tracklets for {}".format(job["my_path"]))
                            loaded_files.append(file)
                    else:
                        terminated.append(job)
                        
                    break
                
        # # intial plot    
        # x_size = 50
        # ratio = x_size / (end_time - start_time) * 40
        # y_size = int((end_x - start_x) /2000*ratio)
        
        # plt.figure(figsize = (x_size,y_size))
        # plt.xlim([start_time,end_time])
        # plt.ylim([start_x,end_x])
        # plt.yticks(fontsize=20)
        # plt.xticks(fontsize=20)
        
        # durations  = plot_tree(tree,trac = 1,msf_dict = msf_dict,COLOR = True)
        # mean_duration = sum(durations)/len(durations)
        # plt.title("{} tracklets with mean duration {:.1f}s".format(len(durations),mean_duration),fontsize = 20)
        
        # plt.savefig("im/{}.png".format(str(plot_counter).zfill(4)))
        # plot_counter += 1
        # plt.show()
        CHANGED = False
        
        # manager timing
        tm = Timer()
        report_time = time.time()
        
        
        # Add jobs to process_queue if dependencies exist
        tm.split("Add jobs")
        # see if any jobs are ready to start
        for item in all_jobs:

            if is_covered(data_dir, item):
                continue
            
            READY = True
            for dependency in item["dep"]:
                if not os.path.exists(dependency):
                    READY = False
                    break
            
            if READY:
                with process_lock:
                    process_queue.put(item)
                all_jobs.remove(item)
        
        
        # spin up worker wrappers
        if False:
            worker_wrapper(data_dir,None,process_queue,process_lock)
        else:
            workers = []
            for i in range(n_workers):
                message_q = mp.Queue()
                # create worker process
                pid = mp.Process(target = worker_wrapper, args = (data_dir,message_q,process_queue,process_lock))
                pid.start()
                workers.append([pid,message_q])
           
        # worker timing
        run_start = time.time()
        time_tracker = {"total":0.1}
        
        while len(terminated) < n_jobs: # TODO get better quitting criteria
            
            
            tm.split("Add jobs")
            # see if any jobs are ready to start
            for item in all_jobs:

                if is_covered(data_dir, item):
                    continue
                
                READY = True
                for dependency in item["dep"]:
                    if not os.path.exists(dependency):
                        READY = False
                        break
                
                if READY:
                    process_queue.put(item)
                    all_jobs.remove(item)
            
            
            
            tm.split("Recieve from queues")
            # listen to each queue for info
            
            # message of length 2 : [job,string]
            # message of length 5: [job, tracklets, tracklets_complete, iteration, time bins]
            
            for s in workers:
                CONTINUE = True
                while CONTINUE: #get every message out of the queue
                    try:
                        data = s[1].get(timeout = 0)
                        if len(data) == 2: # we're going to send messages on the queue as well
                            print("WORKER {}: ".format(s[0].pid) + data[1])
                            
                            if data[1] == "DONE":
                                removals.append(data[0]) # job is added to removals so we can clean up tree and file directory
                            
                        else:
                            
                            #print("Got data from queue")
                            job = data[0] # WILL have to get this info from queue instead now
                            
                            
                            # we have to convert back from numpy to torch...
                            data[1] = [torch.from_numpy(d) for d in data[1]]
                            data[2] = [torch.from_numpy(d) for d in data[2]]
                            
                            # if data[2] >= len(t_thresholds) -2:
                            #     data[2] = "DONE"
                            put_data_in_tree(data[1:4], tree, job["start_time"], job["end_time"], job["start_x"], job["end_x"])
                            CHANGED = True
                            
                            time_data = data[4]
                            if time_data is not None:
                                for key in time_data.keys():
                                    if key in time_tracker.keys():
                                        time_tracker[key] += time_data[key]
                                    else:
                                        time_tracker[key] = time_data[key]

                        
                    except queue.Empty:
                        CONTINUE = False

                        
            
            
            tm.split("Manage process tree")   
            for rm in removals:
                
                terminated.append(rm)
                tree_list = [tree]
                while len(tree_list) > 0:
                    tr = tree_list.pop(0)
                    if rm["start_time"] == tr["start_time"] and rm["end_time"] == tr["end_time"] and rm["start_x"] == tr["start_x"] and rm["end_x"] == tr["end_x"]:
                            
                            #  set data status to done
                            tr["data"][2] = "DONE"
                            tree_list = []
                            print("\n Assigned DONE to node T:{},{} X:{},{}".format(tr["start_time"],tr["end_time"],tr["start_x"],tr["end_x"]))
                            
                            # delete child files to keep data directory tidy
                            if len(tr["children"]) > 0:
                                for ch in tr["children"]:
                                    child_file = "{}/tracklets_{}_{}_{}_{}.cpkl".format(data_dir,ch["start_time"],ch["end_time"],ch["start_x"],ch["end_x"])
                                    print("Successfully deleted child file {}".format(child_file))
                                    try:
                                        os.remove(child_file)
                                    except OSError:
                                        print("WARNING: ----- Failed to delete {}".format(child_file))
                                        
                    else:
                        tree_list += tr["children"]
            removals = []   
                    
                
                
                
            tm.split("Plot and report")
            # only plot if we're not waiting to start new processes
            if CHANGED:
                elapsed = (time.time() - start)
 
                if True:
                    print("\n----------------------------------------------------------------")
                    print("WORKER TIME USAGE:")
                    print("----------------------------------------------------------------")
                    total = time_tracker["total"]
                    for key in time_tracker:
                        print("{}: {:.1f}% ---- {:.1f}s".format(key.ljust(30),time_tracker[key]/total*100,time_tracker[key]))
                    #print("Total time: {:.1f}s [{:.1f}s on main process] for {} processes -- {:.1f}s/process [{:.1f}s main process time]".format(total,total_worker_active_time,n_workers,total/n_workers,elapsed))
                
                    bins = tm.bins()
                    print("\nMANAGER TIME USAGE:")
                    print("----------------------------------------------------------------")
    
                    total = bins["total"]
                    for key in bins:
                        print("{}: {:.1f}% ---- {:.1f}s".format(key.ljust(30),bins[key]/total*100,bins[key]))
                    report_time = time.time()
                    
                    print("\nWorker business: {:.1f}%".format(time_tracker["total"]/(total*n_workers)*100))
                    print("{:.1f}s elapsed, {} / {} terminated chunks".format(time.time() - start,len(terminated),n_jobs))
                
                
                if True:
                    plt.close('all')
                    
                    # 2000 ft = 1 minute
                    
                    x_size = 30
                    ratio = x_size / (end_time - start_time) * 40
                    y_size = int((end_x - start_x) /2000*ratio)
                    
                    plt.figure(figsize = (x_size,y_size))
                    plt.xlim([start_time,end_time])
                    plt.ylim([start_x,end_x])
                    plt.yticks(fontsize=20)
                    plt.xticks(fontsize=20)
                    
                    trac = 100
                    freq = 1
                    if plot_counter % freq == 0:
                        #trac = 1 
                        print("Plotting all tracklets...")
                        
                        durations  = plot_tree(tree,trac = trac,COLOR = True)
                    
                        mean_duration = sum(durations)/len(durations)
                        plt.title("{} tracklets with mean duration {:.1f}s,    {:.1f}s elapsed.".format(len(durations),mean_duration, elapsed),fontsize = 10)
                        
                        if plot_counter % freq == 0:
                            plt.savefig("im/{}.png".format(str(plot_counter).zfill(4)))
                        plt.show()
                    plot_counter += 1

            CHANGED = False                
                
            
        for _ in range(n_workers+2):
            process_queue.put(None)
    
    
    
    except KeyboardInterrupt:
        print("Killing processes....")
    for w in workers:
        w[0].terminate()
        w[0].kill()
        del w[1]
            
    # x_size = 50
    # ratio = x_size / (end_time - start_time) * 40
    # y_size = int((end_x - start_x) /2000*ratio)
    
    if False:
        plt.figure(figsize = (x_size,y_size))
        plt.xlim([start_time,end_time])
        plt.ylim([start_x,end_x])
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        
        durations  = plot_tree(tree,trac = 1,COLOR = True)
        mean_duration = sum(durations)/len(durations)
        plt.title("{} tracklets with mean duration {:.1f}s".format(len(durations),mean_duration),fontsize = 20)
        
        plt.savefig("im/{}.png".format(str(plot_counter).zfill(4)))
        plot_counter += 1
        plt.show() 
    print(("Finished tracking, {:.1f}s elapsed").format(time.time() - start))
    
    from eval_sandbox import evaluate
    gps = "/home/worklab/Documents/datasets/I24-V/final_gps.csv"

    for file in os.listdir(data_dir):
        if "tracklets" in file:
            path = os.path.join(data_dir,file)
            result = evaluate(path,gps)
            mean_duration = int(sum(result["all"]["dur"]) / len(result["all"]["dur"]) * 10)
            save_file = "result_{}".format(mean_duration) + file.split("tracklets")[-1]
            
            save_path = os.path.join(data_dir,save_file)
            with open(save_path,"wb") as f:
                pickle.dump([pass_param_stack,result],f)
            
    # tracklets = get_tracklets(data_dir, tree)
    # tracklets,tracklets_complete = track_chunk(tracklets,tree,pass_params)