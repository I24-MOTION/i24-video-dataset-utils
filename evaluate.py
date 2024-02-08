import numpy as np
import torch
import _pickle as pickle 
from scipy.optimize import linear_sum_assignment
import motmetrics
import pandas as pd

def md_iou(a,b):
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
    union = area_a + area_b - intersection
    iou = torch.div(intersection,union)
    return iou
    #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
    
def get_iou(priors,detections):
       boxes = priors
       d = boxes.shape[0]
       intermediate_boxes = torch.zeros([d,4,2], device = boxes.device)
       intermediate_boxes[:,0,0] = boxes[:,0] 
       intermediate_boxes[:,0,1] = boxes[:,1] - boxes[:,3]/2.0
       intermediate_boxes[:,1,0] = boxes[:,0] 
       intermediate_boxes[:,1,1] = boxes[:,1] + boxes[:,3]/2.0
       
       intermediate_boxes[:,2,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
       intermediate_boxes[:,2,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] - boxes[:,3]/2.0
       intermediate_boxes[:,3,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
       intermediate_boxes[:,3,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] + boxes[:,3]/2.0

       boxes_new = torch.zeros([boxes.shape[0],4],device = boxes.device)
       boxes_new[:,0] = torch.min(intermediate_boxes[:,0:4,0],dim = 1)[0]
       boxes_new[:,2] = torch.max(intermediate_boxes[:,0:4,0],dim = 1)[0]
       boxes_new[:,1] = torch.min(intermediate_boxes[:,0:4,1],dim = 1)[0]
       boxes_new[:,3] = torch.max(intermediate_boxes[:,0:4,1],dim = 1)[0]
       first = boxes_new.clone()
       
       # convert from state form to state-space bbox form
       boxes = detections
       d = boxes.shape[0]
       intermediate_boxes = torch.zeros([d,4,2], device = boxes.device)
       intermediate_boxes[:,0,0] = boxes[:,0] 
       intermediate_boxes[:,0,1] = boxes[:,1] - boxes[:,3]/2.0
       intermediate_boxes[:,1,0] = boxes[:,0] 
       intermediate_boxes[:,1,1] = boxes[:,1] + boxes[:,3]/2.0
       
       intermediate_boxes[:,2,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
       intermediate_boxes[:,2,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] - boxes[:,3]/2.0
       intermediate_boxes[:,3,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
       intermediate_boxes[:,3,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] + boxes[:,3]/2.0

       boxes_new = torch.zeros([boxes.shape[0],4],device = boxes.device)
       boxes_new[:,0] = torch.min(intermediate_boxes[:,0:4,0],dim = 1)[0]
       boxes_new[:,2] = torch.max(intermediate_boxes[:,0:4,0],dim = 1)[0]
       boxes_new[:,1] = torch.min(intermediate_boxes[:,0:4,1],dim = 1)[0]
       boxes_new[:,3] = torch.max(intermediate_boxes[:,0:4,1],dim = 1)[0]
       second = boxes_new.clone()
       
       f = first.shape[0]
       s = second.shape[0]
       
       #get weight matrix
       second = second.unsqueeze(0).repeat(f,1,1).double()
       first = first.unsqueeze(1).repeat(1,s,1).double()
       dist = md_iou(first,second)
       return dist
   
def hmatch(obj_ids,dist,min_match_iou = 0.1):
    s = dist.shape[1]
    try:
        a, b = linear_sum_assignment(dist.data.numpy(),maximize = True) 
    except ValueError:
         return torch.zeros(s)-1
         print("DEREK USE LOGGER WARNING HERE")
     
    
    # convert into expected form
    matchings = np.zeros(s)-1
    for idx in range(0,len(b)):
         matchings[b[idx]] = a[idx]
    matchings = np.ndarray.astype(matchings,int)
     
    # remove any matches too far away
    # TODO - Vectorize this
    for i in range(len(matchings)):
        if matchings[i] != -1 and  dist[matchings[i],i] < min_match_iou:
            matchings[i] = -1    
 
     # matchings currently contains object indexes - swap to obj_ids
    try:
         for i in range(len(matchings)):
            if matchings[i] != -1:
                matchings[i] = obj_ids[matchings[i]]
    except:
        print(type(obj_ids),type(matchings))
        print("Error assigning obj_ids to matchings. len matchings: {}, len obj_ids: {}".format(matchings.shape,obj_ids.shape))
        return torch.zeros(s)-1
                
    return torch.from_numpy(matchings)


def evaluate(gps_path,track_data_path,eval_stride,iou_threshold):
    
    # load data
    with open(track_data_path,"rb") as f:
          tracklets = pickle.load(f)
    print("Loaded tracklet data")
    
    
    # load data
    with open(gps_path,"rb") as f:
          gps = pickle.load(f)
    print("Loaded gps data")
    # gps_obj = {}
    # # ravel gps into a dictionary with x,y,ts,start,end,l,w,h
    # gps = pd.read_csv(gps_path)
    # gps.rename(columns={"Width (ft)":"w","Length (ft)":"l","Height (ft)":"h", "Roadway X (ft)":"x","Roadway Y (ft)":"y","Timestamp (s)":"t"},inplace = True)
    # gps.sort_values(by=["t"])
    
    # for g in range(len(gps)):
        
    #     if g % 1000 == 0:
    #         print(g/len(gps))
    #     row = gps[gps.index == g].squeeze().to_dict()
    #     id = row["id"] 
        
    #     if id in gps_obj.keys():
            
    #         gps_obj[id]["x"].append(row["x"])
    #         gps_obj[id]["y"].append(row["y"])
    #         gps_obj[id]["ts"].append(row["t"])
    #         gps_obj[id]["end"] = row["t"]
    #     else:
    #         gps_obj[id] = {"x":[row["x"]],
    #                        "y":[row["y"]],
    #                        "ts":[row["t"]],
    #                        "start":row["t"],
    #                        "end":row["t"],
    #                        "l":row["l"],
    #                        "w":row["w"],
    #                        "h":row["h"]
                           
    #             }
    
    # gps = gps_obj
    # # iterate over rows and add to associated objects
    # print("Parsed GPS ")

    # container for storing metrics
    metrics = {} 
    metrics["TP"]                =  [0 for _ in gps]
    metrics["GT_total"]          =  [0 for _ in gps]
    metrics["Pred_total"]        =  [0 for _ in tracklets]
    metrics["assigned_ids"]      =  [[] for _ in gps]
    metrics["last_id_assigned"]  =  [None for _ in gps]
    metrics["last_id_count"]     =  [0 for _ in gps]
    metrics["max_n_frames"]      =  [0 for _ in gps]
    metrics["max_distance"]      =  [0 for _ in gps]
    metrics["starting_distance"] =  [0 for _ in gps]
    metrics["MOTP"]              =  []
    metrics["Euclidean"]         =  []
    metrics["HOTA_TP"]           = np.zeros(19)
    metrics["HOTA_grid"]         = np.zeros([19,len(gps),len(tracklets)])
    
    
    # Store start and end times for each tracklet
    
    try:
        with open("dim_cache_{}".format(track_data_path.split("/")[-1]),"rb") as f:
            se,t_dims = pickle.load(f)
    except:
        se = torch.zeros([len(tracklets),2],dtype = torch.double)
        t_dims = torch.zeros([len(tracklets),3])
        #for count,[trk,cls] in enumerate(tracklets):
        for count,trk in enumerate(tracklets):
            trk = trk[0]
            print("\rGetting median for {}/{} tracklets".format(count,len(tracklets)),end = "\r", flush = True)
            se[count,0] = trk[0,0]
            se[count,1] = trk[-1,0]
            
            # stash median dimensions
            l = torch.median(torch.tensor([trk[idx,3] for idx in range(len(trk))]))
            w = torch.median(torch.tensor([trk[idx,4] for idx in range(len(trk))]))
            h = torch.median(torch.tensor([trk[idx,5] for idx in range(len(trk))]))
        
            # # a faster approximation for now
            # l = torch.mean(torch.tensor([trk[idx,3] for idx in range(len(trk))]))
            # w = torch.mean(torch.tensor([trk[idx,4] for idx in range(len(trk))]))
            # h = torch.mean(torch.tensor([trk[idx,5] for idx in range(len(trk))]))
        
            t_dims[count,0] = l
            t_dims[count,1] = w
            t_dims[count,2] = h
        
        with open("dim_cache_{}".format(track_data_path.split("/")[-1]),"wb") as f:
            pickle.dump([se,t_dims],f)
    
    # get min and max of all GPS times
    min_ts = torch.min(se[:,0])
    max_ts = torch.max(se[:,1])
    
    times = np.arange(min_ts,max_ts,step = eval_stride)
    
    # step through all times in time range 
    last_gidx = dict([(gid,0) for gid in gps])
    last_tidx = [0 for _ in tracklets]
    
    tick = 0
    g_converter = {}
    for gid in gps:
        g_converter[gid] = tick
        g_converter[tick] = gid
        tick += 1
        
    
    #%%
    
    for time in times[:100]:
        
        # track progress
        processed = time - times[0]
        total = times[-1] - times[0]
        print("\rInterpolation Progress: {:.1f}/{:.1f}s".format(processed,total),end = "\r",flush = True)
        
        # for each, get the position of all gps objects
        g_include = []
        g_pos = []
        
        for gid in gps:
            if gps[gid]["ts"][0] < time and gps[gid]["ts"][-1] > time:
                g_include.append(gid)
                
                # get position for this time
                while gps[gid]["ts"][last_gidx[gid]] < time:
                    last_gidx[gid] += 1
                
                # interpolate position
                t2 = gps[gid]["ts"][last_gidx[gid]]
                t1 = gps[gid]["ts"][last_gidx[gid]-1]
                x2 = gps[gid]["x"][last_gidx[gid]]
                x1 = gps[gid]["x"][last_gidx[gid]-1]
                r1 = (t2-time)/(t2-t1)
                r2 = 1-r1
                x_int = r1*x1 + r2*x2
                y_int = r1*gps[gid]["y"][last_gidx[gid]-1] + r2*gps[gid]["y"][last_gidx[gid]]
                
                g_pos.append(torch.tensor([x_int,y_int]))
        
    
        # for the set of active objects
        # mask active object set
        
        t_include = (torch.where(se[:,0] < time,1,0) * torch.where(se[:,1] > time,1,0)).nonzero()[:,0]
        t_pos     = []
        for tidx in t_include:
            while tracklets[tidx][0][last_tidx[tidx],0] < time: #last_tidx[tidx]:
                last_tidx[tidx] += 1
                #print("Advanced")
               
            t2 = tracklets[tidx][0][last_tidx[tidx],0]
            t1 = tracklets[tidx][0][last_tidx[tidx]-1,0]
            x2 = tracklets[tidx][0][last_tidx[tidx],1]
            x1 = tracklets[tidx][0][last_tidx[tidx]-1,1]
            y2 = tracklets[tidx][0][last_tidx[tidx],2]
            y1 = tracklets[tidx][0][last_tidx[tidx]-1,2]
            r1 = (t2-time)/(t2-t1)
            r2 = 1-r1
            x_int = r1*x1 + r2*x2
            y_int = r1*y1 + r2*y2
            
            t_pos.append(torch.tensor([x_int,y_int]))
        
    
        # compute distance for each pair or IOU for each pair
        
        if len(t_pos) > 0:
            t_pos = torch.stack(t_pos)
            t_dim = t_dims[t_include,:]
            t_dir = torch.sign(t_pos[:,1]).unsqueeze(1)
            t_pos = torch.cat((t_pos,t_dim,t_dir),dim = 1)
        
        if len(g_pos) > 0:
            g_pos = torch.stack(g_pos)
            g_dim = torch.stack([torch.tensor([gps[gid]["l"],gps[gid]["w"],gps[gid]["h"]]) for gid in g_include])
            g_dir = torch.sign(g_pos[:,1]).unsqueeze(1)
            g_pos = torch.cat((g_pos,g_dim,g_dir),dim = 1)
            
        if len(t_pos) > 0:
            ious = get_iou(g_pos,t_pos)
            #ious = torch.where(ious > iou_threshold,ious,0
        else:
            ious = torch.zeros([len(g_include),1]) + torch.nan
            t_include = [-1]
        
        g_include = torch.tensor([g_converter[gid] for gid in g_include]).unsqueeze(1)
        #g_include = [g_converter[gid.item()] for gid in g_include]
    
        matches = hmatch(g_include,ious)
        
        
        
        # Compute the metrics we care about metrics we need
    
        #### per traj recall = TP / total
        for midx,m in enumerate(matches):
            if m == -1: continue
            gid = m
            tid = t_include[midx]
            
            metrics["TP"][gid] +=1
        
        for gid in g_include:
            metrics["GT_total"][gid] += 1
            
        for tid in t_include:
            metrics["Pred_total"][tid] += 1
            
            
        #### fragmentations per traj - for each trajectory, record list of ids assigned
        for midx,m in enumerate(matches):
            if m == -1: continue
            gid = m
            tid = t_include[midx]
            metrics["assigned_ids"][gid].append(tid.item())
           
        # mean / max fragment length (get directly from tracking data)
        pass
    
        # LCSS - for each fragment trajectory record: last id assigned, num frames assigned, starting distance, max n_frames, max_distance
        for midx,m in enumerate(matches):
            if m == -1: continue
            gid = m
            tid = t_include[midx]
            gidx = (g_include==m).nonzero()[0,0].item()
    
            # this may be wrong, it seems like we should be indexing g_pos with gidx rather than gid
    
            if metrics["last_id_assigned"][gid] == tid:
                metrics["last_id_count"][gid] += 1
                
            else:
                # reset counter
                metrics["last_id_assigned"][gid] = tid
                metrics["starting_distance"][gid] = g_pos[gidx,0]
                metrics["last_id_count"][gid] = 1
            
            # regardless, keep the max counts up to date
            if metrics["last_id_count"][gid] > metrics["max_n_frames"][gid]:
                metrics["max_n_frames"][gid] = metrics["last_id_count"][gid]
                metrics["max_distance"][gid] = torch.abs(metrics["starting_distance"][gid] - g_pos[gidx,0]) # careful, only a sith deals in absolutes
        
        # mean / max trajectory length (Get directly from GPS data)
        pass
        
        for midx,m in enumerate(matches):
            if m == -1: continue
            # we have to use the indices rather than the ids to access the iou and position arrays
            gidx = (g_include==m).nonzero()[0,0].item()
            tidx = midx
            metrics["MOTP"].append(ious[gidx,tidx].item())
            
            # Euclidean distance (Similar to MOTP but distance instead of IOU) -record as list
            dist = torch.sqrt((t_pos[tidx,0] - g_pos[gidx,0])**2 + (t_pos[tidx,0] - g_pos[gidx,0])**2).item()
            metrics["Euclidean"].append(dist)
            
        # modified HOTA  - store an array of trajectory-tracklet pairs and increment every time there is a hit at each threshold
        # at the end, normalize by track and trajectory lengths
        for cidx,cutoff in enumerate(np.arange(0.05,1,step = 0.05)):
            matches = hmatch(g_include,ious,min_match_iou = cutoff)
            for midx,m in enumerate(matches):
                if m == -1: continue
                gid = m
                tid = t_include[midx]
                
                metrics["HOTA_TP"][cidx] += 1
                metrics["HOTA_grid"][cidx,gid,tid] += 1 # fill all cells up to the cutoff cell
            
    
    # Final update on LCSS
        
    
    ###############################################################################
    ########################### Gimme Sympathy and compute some Metrics ###########
    ###############################################################################
        
    with open("metric_cache_{}".format(track_data_path.split("/")[-1]),"wb") as f:
        pickle.dump(metrics,f)
        
    #%% 
    
    with open("metric_cache_{}".format(track_data_path.split("/")[-1]),"rb") as f:
        metrics = pickle.load(f)
    
    # recall
    recall = torch.tensor([metrics["TP"][gid] /(metrics["GT_total"][gid]+0.01) for gid in range(len(gps))])
    recall = recall[recall.nonzero()[:,0]]
    total_recall = torch.mean(recall)
    
    # ids per gt
    ids_per_gt = []
    for id_set in metrics["assigned_ids"]:
        id_set = list(set(id_set))
        if len(id_set) > 0:
            ids_per_gt.append(len(id_set))
    total_ids_per_gt = sum(ids_per_gt)/len(ids_per_gt)
    
    # mean tracking length
    tracklet_durations = torch.tensor([torch.abs(tracklet[0][0,0] - tracklet[0][-1,0]) for tracklet in tracklets])
    tracklet_distances = torch.tensor([torch.abs(tracklet[0][0,1] - tracklet[0][-1,1]) for tracklet in tracklets])
    
    # mean trajectory lengths
    trajectory_durations = torch.tensor([torch.abs(traj["ts"][0] - traj["ts"][-1]) for traj in gps.values()])
    trajectory_distances = torch.tensor([torch.abs(traj["x"][0] - traj["x"][-1]) for traj in gps.values()])
    
    # LCSS -> WRONG!!!
    LCSS = torch.tensor(metrics["max_n_frames"]) * eval_stride
    LCSS_dist = torch.tensor(metrics["max_distance"] )
    mask = (LCSS > 0).int().nonzero()[:,0]
    LCSS = LCSS[mask]
    LCSS_dist = LCSS_dist[mask]
    
    LCSS_dist_total = sum(LCSS_dist)/len(LCSS_dist)
    LCSS_total      = sum(LCSS)/len(LCSS)
    
    # MOTP and Euclidean Distance
    MOTP = sum(metrics["MOTP"])/len(metrics["MOTP"])
    Euclidean = sum(metrics["Euclidean"])/len(metrics["Euclidean"])
    
    # HOTA
    
    ## HOTA_det = HOTA_TP/ GTs      @ each alpha
    HOTA_DET = metrics["HOTA_TP"] / sum(metrics["GT_total"])
    
    # HOTA_ass
    grid = metrics["HOTA_grid"]
    #grid = np.cumsum(grid[::-1,:,:],axis = 0)[::-1,:,:]
    union = torch.zeros([len(gps),len(tracklets)])
    
    # union with intersction double-counted
    for gidx in range(union.shape[0]):
        union[gidx,:] += metrics["GT_total"][gidx]
    for tidx in range(union.shape[1]):
        union[:,tidx] +=  metrics["Pred_total"][tidx]
            
    union = union.unsqueeze(0).expand(grid.shape)
    hota_iou = grid/(union - grid + 1e-05)
    hota_iou = hota_iou.sum(2).sum(1)            
    count_nonzero = np.ceil(grid/100000).sum(2).sum(1)
    HOTA_ASS = hota_iou / count_nonzero
    
    HOTA = (HOTA_ASS * HOTA_DET)**0.5
    mean_HOTA = sum(HOTA)/len(HOTA)
    
    save_metrics = {
            "Recall":recall,
            "Total_Recall":total_recall,
            "IDs":ids_per_gt,
            "Total_IDs":total_ids_per_gt,
            "tracklet_distances":tracklet_distances,
            "tracklet_durations":tracklet_durations,
            "trajectory_durations":trajectory_durations,
            "trajectory_distances":trajectory_distances,
            "MOTP":MOTP,
            "Euclidean":Euclidean,
            "LCSS":LCSS,
            "Total_LCSS":LCSS_total,
            "LCSS_dist":LCSS_dist,
            "Total_LCSS_dist": LCSS_dist_total,
            "HOTA_DET":HOTA_DET,
            "HOTA_ASS":HOTA_ASS,
            "HOTA":HOTA,
            "Total_HOTA":mean_HOTA
            }
    
    with open("{}".format(track_data_path.split("/")[-1]),"wb") as f:
        pickle.dump(save_metrics,f)
        
if __name__ == "__main__":
    
    #gps_path = "/home/worklab/Documents/I24-V/final_gps.csv"
    gps_path = "/home/worklab/Documents/I24-V/track/GPS.cpkl"
    track_path = "/home/worklab/Documents/I24-V/track/results_KIOU_10Hz.cpkl"
    eval_stride = 1/10
    iou_threshold = 0.1
   

    evaluate(gps_path,track_path,eval_stride,iou_threshold)