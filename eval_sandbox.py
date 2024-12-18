import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as pickle
import torch
from evaluate import hmatch, get_iou

colors = np.random.rand(10000,3)
colors[:,0] *= 0.7

if False: 
    
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
        
        
        # plt.plot(raster_times[m1:m2],raster_pos[i,m1:m2,0])
        # plt.plot(t[:,0],t[:,1])
        # plt.show()
    return raster_pos
                

def evaluate(path, gpsf = None,COLLISIONS = False):
    with open(path,"rb") as f:
        tracklets,tracklets_terminated = pickle.load(f) 
        
    path = path.split(".")[0]
    
    
    if gpsf is not None:
        gps = pd.read_csv(gpsf)
        gps = gps.sort_values(by=['Timestamp (s)']).to_numpy()

    
    
    # get start and end time and space
    start_x = int(path.split("_")[3] )
    end_x   = int(path.split("_")[4] )
    start_time = int(path.split("_")[1] )
    end_time = int(path.split("_")[2] )

    try:
        with open("gps_cache_{}".format(path.split("tracklets")[-1]),"rb") as f:
            gps_traj = pickle.load(f)
    except:
        # select relevant GPS
        gps_traj = {}
        if gpsf is not None:
            for ridx,row in enumerate(gps):
                if ridx % 100000 == 0: print("On row {} of {}".format(ridx,len(gps)))
    
                key = row[0]
                ts = row[5]
                x = row[3]
                if  start_x < x and x < end_x and start_time < ts and ts < end_time:
                    if key not in gps_traj.keys():
                        gps_traj[key] = [torch.from_numpy(row[1:].astype(float))]
                    else:
                        gps_traj[key].append(torch.from_numpy(row[1:].astype(float)))
        
        # state_x, state_y,x,y,ts,l,w,h
        for key in gps_traj.keys():
            gps_traj[key] = torch.stack(gps_traj[key])
        
        with open("gps_cache_{}".format(path.split("tracklets")[-1]),"wb") as f:
            pickle.dump(gps_traj,f)
        
    gps = gps_traj # alias
    
        
    
    #################3 Unsupervised
    print(" {} / {} tracklets terminated, {:.1f}%".format(len(tracklets_terminated),len(tracklets) +len(tracklets_terminated), 100*len(tracklets_terminated)/ (len(tracklets) +len(tracklets_terminated))   ))
    
    # for each tracklet, 
        # group it into a lane
    tracklets_all = tracklets + tracklets_terminated
    lane_agg = {1:{"dist":[],"dur":[]},
                 2:{"dist":[],"dur":[]},
                 3:{"dist":[],"dur":[]},
                 4:{"dist":[],"dur":[]},
                 "other":{"dist":[],"dur":[]},
                 "all":{"dist":[],"dur":[]}
                 }
    backwards = []
    forwards = []
    for trk in tracklets_all:    
        
        # get duration in time and space
        duration = (trk[-1,0] - trk[0,0]).item()
        distance = np.abs(trk[-1,1] - trk[0,1]).item()
        lane = int( (torch.mean(trk[:,2]) /-12) -1 )
        
        if lane in lane_agg.keys():
            lane_agg[lane]["dist"].append(distance)
            lane_agg[lane]["dur"].append(duration)
        else:
            lane_agg["other"]["dist"].append(distance)
            lane_agg["other"]["dur"].append(duration)
        lane_agg["all"]["dist"].append(distance)
        lane_agg["all"]["dur"].append(duration)
            
        # note excessive forwards  backwards motion
        diff = trk[1:,1] - trk[:-1,1] 
        tdiff = trk[1:,0] - trk[:-1,0]
        
        test1 = torch.where(diff < -30,1,0)
        test2 = torch.where(tdiff < 0.3,1,0)
        test = test1*test2
        
        maxd = torch.max(diff) # for WB direction, this indicates backwards motion
        if torch.sum(test) > 0:
            forwards.append(torch.sum(test))
                
        bcut = 15
        if maxd > bcut:
            backwards.append(maxd)

        # 200 feet is under 1 second
        # more than 30 feet backwards
    
    print("{} backwards jumps > {} ft., {} forwards jumps > 30 ft in 0.3 second".format(len(backwards),bcut,len(forwards)))    
    print("Durations by lane:")
    for key in lane_agg:
        print("{}: {:.1f}ft, {:.1f}s".format(key,sum(lane_agg[key]["dist"])/len(lane_agg[key]["dist"]), sum(lane_agg[key]["dur"])/len(lane_agg[key]["dur"]) ))
    
    
    # for r in raster_pos:
    #     r = torch.nan_to_num(r,0)
    #     plt.plot(r[:,0])
    # plt.ylim([start_x,end_x])
    # plt.show()
    
    # for each pair look for flips in the sign of raster_pos difference 
    collisions = []
    if COLLISIONS:
        
        # determine whether there are trajectory overlaps
        hz = 0.2
        raster_pos = compute_raster_pos(tracklets_all, start_time, end_time, start_x, end_x,hz = hz)
        
        for i in range(len(raster_pos)-1):
            if i%100 == 0:
                print("On tracklet {}".format(i))
            # for j in range(i+1,len(raster_pos)):
                
            #     diff = raster_pos[i] - raster_pos[j]
            #     diff = torch.nan_to_num(diff,0)
                
            #     sign = torch.sign(diff[:,0])
                
            #     sign_sign = torch.abs( sign[1:] - sign[:-1]) * sign[:-1] # the last bit 0's out all of the nan diff values (sign of nan = 0)
                
            #     sign_sign *= torch.where(torch.abs(raster_pos[i,1] - raster_pos[j,i])[:-1] < 8,1,0) 
                
            #     if torch.max(sign_sign) == 2:
            #         t = torch.argmax(sign_sign)
            #         collisions.append([i,j,start_time+hz*t,raster_pos[i,t,0],raster_pos[j,t,0],raster_pos[i,t,1],raster_pos[j,t,1]])
            #         print(collisions[-1])
            #     #     for t in range(len(sign_sign)):
            #     #         if sign_sign[t] == 2 and torch.abs(raster_pos[i,t,1] - raster_pos[j,t,1]) < 8: # in same lane and swap positions
            #     #             collisions.append([i,j,start_time+hz*t,raster_pos[i,t,0],raster_pos[j,t,0],raster_pos[i,t,1],raster_pos[j,t,1]])
            #     #             #break # we just record the first
               
            collision_dist = 9
            ras = raster_pos[i].unsqueeze(0).expand(raster_pos.shape[0],raster_pos.shape[1],2)
            others = raster_pos
            assert ras.shape == others.shape
            
            diff = torch.nan_to_num(ras - others,0)
            sign = torch.sign(diff[:,:,0])
            sign_sign = torch.abs( sign[:,1:] - sign[:,:-1])
            sign_sign *= torch.where(torch.abs(ras[:,:-1,1] - others[:,:-1,1]) < collision_dist,1,0)
            
            intersections = torch.where(sign_sign == 2,1,0).nonzero()
            
            for item in intersections:
                j = item[0]
                t = item[1]
                collisions.append([i,j,start_time+hz*t,raster_pos[i,t,0],raster_pos[j,t,0],raster_pos[i,t,1],raster_pos[j,t,1]])
            # for j in range(sign_sign.shape[0]):
            #     for t in range(sign_sign.shape[1]):
            #         if sign_sign[j,t] == 2:
            #             collisions.append([i,j,start_time+hz*t,raster_pos[i,t,0],raster_pos[j,t,0],raster_pos[i,t,1],raster_pos[j,t,1]])
                      
                        
                
                
                
        #[print(c) for c in collisions]
        unique = list(set([c[0] + c[1] for c in collisions]))
        print("{} unique trajectory conflicts".format(len(unique)))
    
    SHOW = True
    if SHOW:
        for lane in [1,2,3,4,5.25]:
             plt.figure(figsize = (60,30))
             for tidx,tra in enumerate(tracklets_all):
                 mask = torch.where(torch.logical_and(tra[:,2]  < -12*(lane), tra[:,2] > -12*(lane+1)),1,0).nonzero().squeeze(1)
                 if len(mask) < 2: continue 
                 t = tra[mask,:]
                 #if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
                 plt.plot(t[:,0],t[:,1],c = colors[tidx],linewidth = 4)
                 plt.text(t[0,0],t[0,1],tidx)
             # for tidx,t in enumerate(tracklets_terminated):
             #     if t[0,2] < -12*(lane) and t[0,2] > -12*(lane+1):
             #         plt.plot(t[:,0],t[:,1],c = "b",linewidth = 4)
             #         plt.text(t[0,0],t[0,1],tidx)

                     
             for c in collisions:
                 if c[5] < -12*(lane) and c[5] > -12*(lane+1):
                     plt.plot([c[2],c[2]],c[3:5],marker = "x",c = "r",markersize = 20)
                     #plt.text(c[2],c[3],"{}:{}".format(c[0],c[1]))
            
     
             for key in gps_traj.keys():
                  tra = gps_traj[key]
                  mask = torch.where(torch.logical_and(tra[:,3]  < -12*(lane), tra[:,3] > -12*(lane+1)),1,0).nonzero().squeeze(1)
                  if len(mask) < 2: continue 
                  t = tra[mask,:]
                  
                  plt.plot(t[:,4],t[:,2],":",color = (0,0,0),linewidth = 8)
             

             plt.title("Post clustering lane {}".format(lane))
             plt.show()
             
    ##################### Supervised
    tracklets = tracklets_all


    # slice relevant portions of GPS trajectory
    
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
    
    
    # Store start and end times for each tracklet as well as dimensions
    
    try:
        with open("dim_cache_{}".format(path.split("tracklets")[-1]),"rb") as f:
            se,t_dims = pickle.load(f)
    except:
        se = torch.zeros([len(tracklets),2],dtype = torch.double)
        t_dims = torch.zeros([len(tracklets),3])
        #for count,[trk,cls] in enumerate(tracklets):
        for count,trk in enumerate(tracklets):
            
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
        
        with open("dim_cache_{}".format(path.split("tracklets")[-1]),"wb") as f:
            pickle.dump([se,t_dims],f)




    
    eval_stride = 0.1
    times = np.arange(start_time,end_time,step = eval_stride)
    
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
    
    # gps is    [key] : state_x, state_y,x,y,ts,l,w,h
    
    for time in times:
        
        # track progress
        processed = time - times[0]
        total = times[-1] - times[0]
        print("\rInterpolation Progress: {:.1f}/{:.1f}s".format(processed,total),end = "\r",flush = True)
        
        # for each, get the position of all gps objects
        g_include = []
        g_pos = []
        
        for gid in gps:
            if gps[gid][0,4] < time and gps[gid][-1,4] > time:
                
                
                # get position for this time
                while gps[gid][4,last_gidx[gid]] < time:
                    last_gidx[gid] += 1
                
                # interpolate position
                t2 = gps[gid][last_gidx[gid],4]
                t1 = gps[gid][last_gidx[gid]-1,4]
                x2 = gps[gid][last_gidx[gid],2]
                x1 = gps[gid][last_gidx[gid]-1,2]
                r1 = (t2-time)/(t2-t1)
                r2 = 1-r1
                x_int = r1*x1 + r2*x2
                y_int = r1*gps[gid][last_gidx[gid]-1,3] + r2*gps[gid][last_gidx[gid],3]
                
                if x_int > start_x and x_int < end_x and torch.sign(y_int) == -1:
                    g_pos.append(torch.tensor([x_int,y_int]))
                    g_include.append(gid)
    
        # for the set of active objects
        # mask active object set
        
        t_include = (torch.where(se[:,0] < time,1,0) * torch.where(se[:,1] > time,1,0)).nonzero()[:,0]
        t_pos     = []
        for tidx in t_include:
            while tracklets[tidx][last_tidx[tidx],0] < time: #last_tidx[tidx]:
                last_tidx[tidx] += 1
                #print("Advanced")
               
            t2 = tracklets[tidx][last_tidx[tidx],0]
            t1 = tracklets[tidx][last_tidx[tidx]-1,0]
            x2 = tracklets[tidx][last_tidx[tidx],1]
            x1 = tracklets[tidx][last_tidx[tidx]-1,1]
            y2 = tracklets[tidx][last_tidx[tidx],2]
            y1 = tracklets[tidx][last_tidx[tidx]-1,2]
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
            g_dim = torch.stack([torch.tensor([gps[gid][0,5],gps[gid][0,6],gps[gid][0,7]]) for gid in g_include])
            g_dir = torch.sign(g_pos[:,1]).unsqueeze(1)
            g_pos = torch.cat((g_pos,g_dim,g_dir),dim = 1)
        else:
            g_pos = torch.empty([0,6])
            
            
            
            
        if len(t_pos) > 0 and len(g_pos) > 0:
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
        
    with open("metric_cache_{}".format(path.split("tracklets")[-1]),"wb") as f:
        pickle.dump(metrics,f)
        
    #%% 
    
    with open("metric_cache_{}".format(path.split("tracklets")[-1]),"rb") as f:
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
    total_ids_per_gt = sum(ids_per_gt)/(len(ids_per_gt) +0.0001)
    
    # mean tracking length
    tracklet_durations = torch.tensor([torch.abs(tracklet[0,0] - tracklet[-1,0]) for tracklet in tracklets])
    tracklet_distances = torch.tensor([torch.abs(tracklet[0,1] - tracklet[-1,1]) for tracklet in tracklets])
    
    # mean trajectory lengths
    trajectory_durations = torch.tensor([torch.abs(traj[0,4] - traj[-1,4]) for traj in gps.values()])
    trajectory_distances = torch.tensor([torch.abs(traj[0,2] - traj[-1,2]) for traj in gps.values()])
    
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
    print(save_metrics)
    
    
    # find closest trajectory matches - we can largely reuse code from previous iteration
    
    
    results = lane_agg
    results["collisions"] = collisions
    results["backwards"]  = backwards
    results["forwards"]   = forwards
    
    for key in save_metrics.keys():
        results[key] = save_metrics[key]
        
    
    return results
    
    
if __name__ == "__main__":   
    
    path = "/home/worklab/Documents/i24/i24-video-dataset-utils/data/1/tracklets_0_900_7000_10000.cpkl"
    path = "/home/worklab/Documents/i24/i24-video-dataset-utils/data/bm2/tracklets_1000_1060_15000_18000.cpkl"
    gps = "/home/worklab/Documents/datasets/I24-V/final_gps.csv"
    result = evaluate(path,gps)