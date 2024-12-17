import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as pickle
import torch

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
    
    # slice relevant portions of GPS trajectory
    
    
    # find closest trajectory matches - we can largely reuse code from previous iteration
    
    
    results = lane_agg
    results["collisions"] = collisions
    results["backwards"]  = backwards
    results["forwards"]   = forwards
        
    
    return results
    
    
if __name__ == "__main__":   
    
    path = "/home/worklab/Documents/i24/i24-video-dataset-utils/data/1/tracklets_0_900_7000_10000.cpkl"
    path = "/home/worklab/Documents/i24/i24-video-dataset-utils/data/bm2/tracklets_1000_1060_15000_18000.cpkl"
    gps = "/home/worklab/Documents/datasets/I24-V/final_gps.csv"
    evaluate(path,gps)