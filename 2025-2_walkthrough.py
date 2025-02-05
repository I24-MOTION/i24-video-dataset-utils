if __name__ == "__main__":
    # imports
    import numpy as np
    import torch
    import _pickle as pickle
    import pandas as pd
    from i24_rcs import I24_RCS  
    import matplotlib.pyplot as plt
    import cv2
    colors = np.random.rand(100000,3)
    
    
    
    # load each file
    
    #reference images 
    reference_dir = "/home/worklab/Documents/datasets/I24-V/reference"
    
    # homography
    hg_cache_file = "/home/worklab/Documents/datasets/I24-V/WACV2024_hg_save.cpkl"
    hg_directory  = "/home/worklab/Documents/datasets/I24-V/wacv_hg_v1"
    
    # time-indexed data
    detection_path = "/home/worklab/Documents/datasets/I24-V/final_detections.npy"
    old_track_path = "/home/worklab/Documents/datasets/2025-2-gamma-tracking/MOTION_1HR_OLD.npy"
    new_track_path = "/home/worklab/Documents/datasets/2025-2-gamma-tracking/MOTION_1HR_NEW.npy"
    gps_data_path = "/home/worklab/Documents/datasets/I24-V/final_gps.csv"
    
    
    # trajectory-indexed data
    old_trajectory_path = "/home/worklab/Documents/i24/cluster-track-dev/data/0/post_0_3600_4000_20000.cpkl"
    new_trajectory_path = "/home/worklab/Documents/i24/cluster-track-dev/data/0/tracklets_0_3600_4000_20000.cpkl"
    
    
    path = new_trajectory_path
    
    
    #%% Parse time-indexed GPS data into GPS trajectories
    gps = pd.read_csv(gps_data_path)
    gps = gps.sort_values(by=['Timestamp (s)']).to_numpy()
    
    
    
    # get start and end time and space
    start_x = int(path.split(".")[0].split("_")[3] )
    end_x   = int(path.split(".")[0].split("_")[4] )
    start_time = int(path.split(".")[0].split("_")[1] )
    end_time = int(path.split(".")[0].split("_")[2] )
    
    try:
        with open("gps_trajectories.cpkl","rb") as f:
            gps_traj = pickle.load(f)
    except:
        print("Parsing GPS time-indexed data into trajectories...")
        # select relevant GPS
        gps_traj = {}
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
        
        with open("gps_trajectories.cpkl","wb") as f:
            pickle.dump(gps_traj,f)
                
                
    
    #%% Do some stuff with trajectory-indexed data
    
    try:
        with open(path,"rb") as f:
            tracklets,tracklets_terminated = pickle.load(f) 
    except:
        with open(path,"rb") as f:
            tracklets = pickle.load(f) 
            tracklets_terminated = []
           
    tracklets_all = tracklets + tracklets_terminated
    
    # for storing metrics
    lane_agg = {1:{"dist":[],"dur":[]},
                  2:{"dist":[],"dur":[]},
                  3:{"dist":[],"dur":[]},
                  4:{"dist":[],"dur":[]},
                  "other":{"dist":[],"dur":[]},
                  "all":{"dist":[],"dur":[]}
                  }
    backwards = []
    forwards = []
    
    
    print("\nComputing length and duration statistics for trajectory data...")
    for trk in tracklets_all:    
        
        if len(trk) < 3:
            continue
        
        
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
        print("{}: {:.1f}ft, {:.1f}s".format(key,sum(lane_agg[key]["dist"])/(0.1+len(lane_agg[key]["dist"])), sum(lane_agg[key]["dur"])/(0.1+len(lane_agg[key]["dur"]) )))
    
    
    #%% PLot a selection of trajectory data
    # pick window to plot 
    xmin = 8000
    xmax = 13000
    tmin = 500
    tmax = 1000
    
    print("Plotting trajectory data...")
    for lane in [1,2,3,4,5.25]:
          plt.figure(figsize = (40,20))
          for tidx,tra in enumerate(tracklets_all):
              mask = torch.where(torch.logical_and(tra[:,2]  < -12*(lane), tra[:,2] > -12*(lane+1)),1,0)
              mask2 = torch.where(torch.logical_and(tra[:,1]  < xmax, tra[:,1] > xmin),1,0)
              mask3 = torch.where(torch.logical_and(tra[:,0]  < tmax, tra[:,0] > tmin),1,0)
              mask = (mask * mask2 * mask3).nonzero().squeeze(1)
              if len(mask) < 2: continue 
              t = tra[mask,:]
              plt.plot(t[:,0],t[:,1],c = colors[tidx],linewidth = 2)
              plt.text(t[0,0],t[0,1],tidx)
    
     
          for key in gps_traj.keys():
              tra = gps_traj[key]
              mask = torch.where(torch.logical_and(tra[:,3]  < -12*(lane), tra[:,3] > -12*(lane+1)),1,0)
              mask2 = torch.where(torch.logical_and(tra[:,2]  < xmax, tra[:,2] > xmin),1,0)
              mask3 = torch.where(torch.logical_and(tra[:,4]  < tmax, tra[:,4] > tmin),1,0)
              mask = (mask * mask2 * mask3).nonzero().squeeze(1)
              if len(mask) < 2: continue 
              t = tra[mask,:]
              
              plt.plot(t[:,4],t[:,2],":",color = (0.5,0.5,0.5),linewidth = 5)
              plt.text(t[0,4],t[0,2],key)
    
    
          plt.title("Tracklets lane {}".format(lane),fontsize = 24)
          plt.xlabel("Time (s)",fontsize = 24)
          plt.ylabel("X-position (ft)",fontsize = 24)
          plt.show()
    
    
    
    #%% work with coordinate system
    
    rcs = I24_RCS(hg_cache_file,downsample = 2, default = "static")
    rcs.load_correspondences_WACV(hg_directory)
    
#%%    
    try:
        with open(path,"rb") as f:
            tracklets,tracklets_terminated = pickle.load(f) 
    except:
        with open(path,"rb") as f:
            tracklets = pickle.load(f) 
            tracklets_terminated = []
           
    
    #%%  MOTION data is stored in the roadway coordinate frame -- let's convert it to State Plane coordinates
    # %matplotlib (interactive backend) is recommended
    
    count = 500
    plt.figure(figsize = (10,10))
    print("Plotting state plane data (epsg:2274) for first {} tracklets".format(count))
    
    for tidx in range(count):
        tracklet = tracklets[tidx]
        
        # currently, the data is of the form [n_observations, 10] with each row [time,x_pos,y_pos,length,width,height,class, confidence, <metadata>,<metadata>]
        
        # the coordinate system expects data in the form [n,6] with each row [xpos,ypos,length,width,height,direction of travel]
        data = torch.cat((tracklet[:,[1,2,3,4,5]],torch.sign(tracklet[:,2:3])),dim = 1)
        
        # put data in form [n,8,3] giving 8 bounding box corner coordinates for each bounding box
        state_plane_data = rcs.state_to_space(data)
        
        # for simplicity, we plot only the first corner of each bounding box position
        plt.scatter(state_plane_data[:,0,0],state_plane_data[:,0,1])
    
    plt.xlabel("State plane X (ft)")
    plt.ylabel("State plane Y (ft)")
    plt.show()
    
    #%% Similarly, we can convert to GPS coordinate system as well
    # %matplotlib (interactive backend) is recommended
    
    count = 500
    plt.figure(figsize = (10,10))
    print("Plotting GPS data (epsg:4326) for first {} tracklets".format(count))
    
    for tidx in range(count):
        tracklet = tracklets[tidx]
        
        # currently, the data is of the form [n_observations, 10] with each row [time,x_pos,y_pos,length,width,height,class, confidence, <metadata>,<metadata>]
        
        # the coordinate system expects data in the form [n,6] with each row [xpos,ypos,length,width,height,direction of travel]
        data = torch.cat((tracklet[:,[1,2,3,4,5]],torch.sign(tracklet[:,2:3])),dim = 1)
        
        # put data in form [n,8,3] giving 8 bounding box corner coordinates for each bounding box
        state_plane_data = rcs.state_to_space(data)
        gps_data = rcs.space_to_gps(state_plane_data)
        plt.scatter(gps_data[:,0],gps_data[:,1])
    
    plt.xlabel("GPS Lat")
    plt.ylabel("GPS Long")
    plt.show()
    
    
    #%% get extents for each camera FOV from hg_save_file
    # this is nice if you want to plot vehicle positions in a camera field of view  
    # in practice it makes sense to check for inclusion in the X range and simple sign matching in the Y range since the area covered by a camera extends in the Y direction somewhat beyond the dashed lane markings
    
           
           
    camera_extents = {}
    for key in rcs.correspondence.keys():
        corr = rcs.correspondence[key]
        dash_pts = corr["space_pts"] # labeled dash line points visible in this camera, in state plane coordinates
        
        # put into expected form
        dash_pts = torch.tensor(dash_pts)
        dash_pts = torch.cat((dash_pts,torch.zeros(dash_pts.shape[0],1)),dim = 1).unsqueeze(1)
        state_dashes = rcs.space_to_state(dash_pts)
        xmin = torch.min(state_dashes[:,0])
        xmax = torch.max(state_dashes[:,0])
        ymin = torch.min(state_dashes[:,1])
        ymax = torch.max(state_dashes[:,1])
        camera_extents[key] = [xmin,ymin,xmax,ymax]
        
    # for sanity, plot a few bounding boxes in a camera image
    
    tidx = 101
    tracklet = tracklets[tidx]
    cv2.namedWindow("window")
    
    count = 0
    buf = 50
    buffer = None
    
    
    for pos in tracklet:
        
        # get first relevant camera
        
        for corr in camera_extents.keys():
            ext = camera_extents[corr]
            
            if pos[1] > ext[0] and pos[1] < ext[2] and torch.sign(pos[2]) == torch.sign(ext[1]):
                break
    
        # plot box in camera
        im = cv2.imread(reference_dir + "/1080p/{}.png".format(corr.split("_")[0]))
        box = torch.tensor([pos[1],pos[2],pos[3],pos[4],pos[5],torch.sign(pos[2])]).unsqueeze(0)
        
    
        if buffer is not None:
            im = rcs.plot_state_boxes(im, buffer,name = corr,color = (200,200,200),thickness = 1)
        im = rcs.plot_state_boxes(im, box,name = corr,color = (0,50,200),thickness = 2)
    
        if buffer is None:
            buffer = box
        else:
            buffer = torch.cat((buffer,box),dim = 0)[-buf:,:]
    
        cv2.putText(im, corr, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),thickness = 3)                
        cv2.putText(im, corr, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0))                


        cv2.imshow("window",im)
        cv2.setWindowTitle("window","{:.3f}s".format(pos[0]))
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        
        #cv2.imwrite("im/{}.png".format(str(count).zfill(5)),im)
        count += 1
        
    cv2.destroyAllWindows()
    
        
    #%% run overhead viewer
    from viz_detections import Viewer 
    
    #det = np.load(detection_path)
    det = np.load(new_track_path)
    #gps = pd.read_csv(gps_data_path)
    gps = None
    
    v = Viewer(det,gps)
    v.run()
        
    
    
    
    #%% Plot boxes for one camera
    if __name__ == "__main__":
        from data_viewer import VideoViewer
    
        video_dir = "/home/worklab/Documents/datasets/I24-V/video"
        camera_names   = ["P32C02","P32C04","P32C05","P32C06"]
           
        dv = VideoViewer(video_dir,
                        camera_names,
                        rcs,
                        buffer_frames = 400,
                        start_time = 0, 
                        gps = None,
                        manual = None,
                        detections = new_track_path)
        dv.run()
          
                
      
        
    
        
    
    
    
    
