if __name__ == "__main__":
    # imports
    import numpy as np
    import torch
    import _pickle as pickle
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2
    colors = np.random.rand(100000,3)
    
    # import custom repo, use: pip3 install git+https://github.com/I24-MOTION/i24_rcs@v1.1-stable
    from i24_rcs import I24_RCS  

    
    # load each file
    
    #reference images 
    reference_dir = "/home/worklab/Documents/datasets/I24-V/reference"
    
    # homography
    hg_cache_file = "/home/worklab/Documents/datasets/I24-V/WACV2024_hg_save.cpkl"
    hg_directory  = "/home/worklab/Documents/datasets/I24-V/wacv_hg_v1"                  
    
    rcs = I24_RCS(hg_cache_file,downsample = 2, default = "static")
    rcs.load_correspondences_WACV(hg_directory) 
    
    # time-indexed data
    detection_path = "/home/worklab/Documents/datasets/I24-V/final_detections.npy"                    # each row is [time,x pos,y pos,length,width,height,veh class,detector confidence]
    old_track_path = "/home/worklab/Documents/datasets/2025-2-gamma-tracking/MOTION_OLD_TIME.npy"     # each row is [time,x pos,y pos,length,width,height,veh class,id]
    new_track_path = "/home/worklab/Documents/datasets/2025-2-gamma-tracking/MOTION_NEW_TIME.npy"     # each row is [time,x pos,y pos,length,width,height,veh class,id]
    gps_data_path = "/home/worklab/Documents/datasets/I24-V/final_gps.csv"                            # each row is [id,state plane x, state plane y,x,y,ts,length, width, height]
    
    
    # trajectory-indexed data
    old_trajectory_path = "/home/worklab/Documents/datasets/2025-2-gamma-tracking/old_0_3600_4000_20000.cpkl"     # dictionary indexed by id, each element is array with rows [time,x,y,l,w,h,class,ignore,ignore]]
    new_trajectory_path = "/home/worklab/Documents/datasets/2025-2-gamma-tracking/new_0_3600_4000_20000.cpkl"     # dictionary indexed by id, each element is array with rows [time,x,y,l,w,h,class,ignore,ignore]]
    
    path = new_trajectory_path # alias to old or new path
    
    
    def load_MOTION_traj(file):
        try:
            with open(path,"rb") as f:
                tracklets,tracklets_terminated = pickle.load(f) 
        except:
            with open(path,"rb") as f:
                tracklets = pickle.load(f) 
                tracklets_terminated = []
               
        return tracklets + tracklets_terminated    
    
    def traj_to_tensor(traj):
        t = torch.zeros(len(traj["x_position"]),6)
        t[:,0] = torch.tensor(traj["timestamp"])
        t[:,1] = torch.tensor(traj["x_position"])
        t[:,2] = torch.tensor(traj["y_position"])
        t[:,3] = traj["length"]
        t[:,4] = traj["width"]
        t[:,5] = traj["height"]
        return t
    
    
    #%% 1. Working with the I-24 coordinate systems
    # %matplotlib (interactive backend) is recommended


    
    
    # create a grid of dummy data along the road
    # Expected form is a tensor of size [n_objects,6] 
    # x_position (feet), y_position, length, width, height, direction (1 for EB or -1 for WB)
    xrange = torch.arange(0,25000,step = 100)
    yrange = torch.arange(-60,60,step = 12)
    
    
    xrange = xrange.unsqueeze(0).expand(yrange.shape[0],xrange.shape[0])
    yrange = yrange.unsqueeze(1).expand(xrange.shape)
    xrange = xrange.reshape(-1)
    yrange = yrange.reshape(-1)
    
    plt.figure()
    plt.scatter(xrange,yrange)
    for idx in range(xrange.shape[0]):
        plt.text(xrange[idx],yrange[idx],"{}ft,{}ft".format(xrange[idx],yrange[idx]),rotation = 45)
    plt.xlabel("Roadway X (ft)")
    plt.ylabel("Roadway Y (ft)")
    plt.show()
    
    
    l = torch.zeros(xrange.shape)
    w = torch.zeros(xrange.shape)
    h = torch.zeros(xrange.shape)
    direction = torch.sign(yrange)
    
    # [n_pts,6]
    roadway_pts = torch.stack([xrange,yrange,l,w,h,direction]).transpose(1,0)

    xrange = xrange.data.numpy()
    yrange = yrange.data.numpy()
    #[n_pts,8,3] 
    # middle dimension is for corner coordiantes of a bounding box - since we set l,w,h to 0, we can use any corner index
    # last dimension indexes state plane x, state plane y, z coordinate
    state_plane_points = rcs.state_to_space(roadway_pts)
    
    # plot the grid in state plane
    plt.figure()
    plt.scatter(state_plane_points[:,0,0],state_plane_points[:,0,1])
    for idx in range(state_plane_points.shape[0]):
        plt.text(state_plane_points[idx,0,0],state_plane_points[idx,0,1],"{}ft,{}ft".format(xrange[idx],yrange[idx]))

    plt.xlabel("State plane X (ft)")
    plt.ylabel("State plane Y (ft)")
    plt.axis("square")
    plt.show()
    
    
    
    #%%  1b. MOTION data is stored in the roadway coordinate frame -- let's convert it to State Plane coordinates
    # %matplotlib (interactive backend) is recommended
    tracklets = load_MOTION_traj(path)


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
    
    #%% 1c. Similarly, we can convert to GPS coordinate system as well
    # %matplotlib (interactive backend) is recommended
    tracklets = load_MOTION_traj(path)


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
    
    
    #%% 2. Parse time-indexed GPS data into GPS trajectories 
    # This converts the gps csv file into a similar format to the trajectory-indexed data files above
    
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
        
        # cache a save file so this doesn't need to be run again
        with open("gps_trajectories.cpkl","wb") as f:
            pickle.dump(gps_traj,f)
                
                
    
    #%% 3. Compute aggregate data statistics with the trajectory-indexed data
    # you can run this for either the old data or new data - modify path accordingly above
    
    tracklets = load_MOTION_traj(path)
    
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
    for trk in tracklets:    
        
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
    
    
    #%% 4. Plot a selection of trajectory data 
    tracklets = load_MOTION_traj(path)


    # pick window to plot 
    xmin = 4000
    xmax = 10000
    tmin = 500
    tmax = 800
    
    print("Plotting trajectory data...")
    for lane in [1,2,3,4,5.25]:
          plt.figure(figsize = (40,20))
          for tidx,tra in enumerate(tracklets):
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
    
    
    

    


    
    
    #%% 5. get extents for each camera FOV from hg_save_file
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
        print ("{} covers X range: [{:.1f},{:.1f}]ft;  Y range: [{:.1f},{:.1f}]ft".format(key,xmin,xmax,ymin,ymax))
    # for sanity, plot a few bounding boxes in a camera image



    #%% 5b. Plot bounding boxes for a trajectory in background images
    tracklets = load_MOTION_traj(path)


    tidx = 100
    traj = tracklets[tidx]
    cv2.namedWindow("window")
    
    count = 0
    buf = 50
    buffer = None
    
    
    from utils_opt import resample,opt2_l1_constr
    lam1_x= 3e-1
    lam2_x= 0
    lam3_x= 1e-7

    lam1_y= 0
    lam2_y= 0
    lam3_y= 1e-3

    
    # wrangle form for resampling
    car = {"timestamp" : traj[:,0].data.numpy(),
           "x_position": traj[:,1].data.numpy(),
           "y_position": traj[:,2].data.numpy(),
           "direction" : torch.sign(traj[0,2]).item(),
           "length":traj[0,3],
           "width" :traj[0,4],
           "height":traj[0,5]
           }
        
    re_car = resample(car,dt = 0.04,fillnan = True)
    smooth_car = opt2_l1_constr(re_car.copy(), lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y)
    re_car = traj_to_tensor(re_car)
    tracklet = traj_to_tensor(smooth_car)
    
    
    
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
    
        
    #%% 6. Run overhead viewer
    from viz_detections import Viewer 
    
    # this can be the detection_path or either the new or old track path
    #det = np.load(detection_path)
    det = np.load(new_track_path)
    
    gps = pd.read_csv(gps_data_path)
    
    
    v = Viewer(det,gps)
    v.run()
    
    # a few key controls
      #   # (shift 3) - load presaved parameters
      #   < > - move forward and backwards on roadway
      #   [ ] - adjust y range plotted
      #   - + - zoom in and out
      #   q w - go backwards / forwards in time
      #   963 - activate the x,y,z vanishing points - you can then click and drag to move those vanishing points
 

    
    #%% 8. Lets apply some smoothing as from: https://github.com/I24-MOTION/I24-postprocessing
    # if you reuse this code, please cite Yanbing Wang's paper "Automatic vehicle trajectory data reconstruction at scale" from which the method is taken
    
    from utils_opt import resample,opt2_l1_constr
    lam1_x= 3e-1
    lam2_x= 0
    lam3_x= 1e-7
    
    lam1_y= 0
    lam2_y= 0
    lam3_y= 1e-3
    
    
    
    
    # select a trajectory 
    tidx = 100
    tracklets = load_MOTION_traj(path)
    traj = tracklets[tidx]
    
    # we need to put this trajectory back into the original I-24 MOTION inception format
    # dictionary with at least x_position,y_position,timestamp,direction   
    car = {"timestamp" : traj[:,0].data.numpy(),
           "x_position": traj[:,1].data.numpy(),
           "y_position": traj[:,2].data.numpy(),
           "direction" : torch.sign(traj[0,2]).item(),
           "length":traj[0,3],
           "width" :traj[0,4],
           "height":traj[0,5]
           }
        
    re_car = resample(car,dt = 0.04,fillnan = True)
    smooth_car = opt2_l1_constr(re_car.copy(), lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y)
    
    re_car = traj_to_tensor(re_car)
    smooth_car = traj_to_tensor(smooth_car)
    
    
    plt.figure()
    plt.plot(traj[:,0],traj[:,1])
    plt.plot(re_car[:,0],re_car[:,1])
    plt.plot(smooth_car[:,0],smooth_car[:,1])
    plt.ylabel("X-Position (ft)")
    plt.xlabel("Time (s)")
    plt.legend(["start","resample","smooth"])
    
    plt.figure()
    plt.plot(traj[:-1,0],(traj[1:,1]-traj[:-1,1])/(traj[1:,0]-traj[:-1,0]))
    plt.plot(re_car[:-1,0],(re_car[1:,1]-re_car[:-1,1])/(re_car[1:,0]-re_car[:-1,0]))
    plt.plot(smooth_car[:-1,0],(smooth_car[1:,1]-smooth_car[:-1,1])/(smooth_car[1:,0]-smooth_car[:-1,0]))
    plt.ylim([-150,150])
    plt.ylabel("X-Velocity (ft/s)")
    plt.xlabel("Time (s)")
    plt.legend(["start","resample","smooth"])
    
    
    #%% 9. Lets smooth a platoon
    # if you reuse this code, please cite Yanbing Wang's Paper "Automatic vehicle trajectory data reconstruction at scale" from which the method is taken
    
    
    from utils_opt import resample,opt2_l1_constr
    lam1_x= 3e-1
    lam2_x= 0
    lam3_x= 1e-7
    
    lam1_y= 0
    lam2_y= 0
    lam3_y= 1e-3
    
    
    
    tracklets = load_MOTION_traj(path)
    
    
    # you can select another set of ids manually by inspecting the above time-space plots
    traj_ids = [46787,46614,46710,46796,1326,46981,46689,46653,674,46657,46728,46674,46616]
    #traj_ids = [46787]
    
    
    
    
    
    smoothed = []
    
    # iterate over trajectories
    for t,tidx in enumerate(traj_ids):
        print("On tracklet {}".format(tidx))
        traj = tracklets[tidx]
        
        # wrangle form for resampling
        car = {"timestamp" : traj[:,0].data.numpy(),
               "x_position": traj[:,1].data.numpy(),
               "y_position": traj[:,2].data.numpy(),
               "direction" : torch.sign(traj[0,2]).item(),
               "length":traj[0,3],
               "width" :traj[0,4],
               "height":traj[0,5]
               }
            
        re_car = resample(car,dt = 0.04,fillnan = True)
        smooth_car = opt2_l1_constr(re_car.copy(), lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y)
        re_car = traj_to_tensor(re_car)
        smooth_car = traj_to_tensor(smooth_car)
        smoothed.append([t,tidx,smooth_car])
    
    #%% Plot the result
    
    # define colors for plotting
    c = np.linspace(0,1,num = len(traj_ids))
    colors = np.zeros([len(traj_ids),3])
    colors[:,0] = c
    colors[:,2] = 1-c
    #colors = np.random.rand(30,3)
        
    
    fig, ax = plt.subplots(3,1, sharex=True)
    leg = []
    for packet in smoothed:
        t,tidx,smooth_car = packet
        # plot position
        ax[0].plot(smooth_car[:,0],smooth_car[:,1],color = colors[t])
        
        
        # plot relative position relative to lead vehicle
        lead_idx = 0
        
        rel_x = []
        rel_xt = []
        for idx in range(smooth_car.shape[0]):
            while smooth_car[idx,0] < smoothed[0][2][0,0]:
                continue # skip all times before the lead car exists
                
            while lead_idx < smoothed[0][2].shape[0] and smoothed[0][2][lead_idx,0] < smooth_car[idx,0]:
                lead_idx += 1
            
            if lead_idx >= smoothed[0][2].shape[0]: break # skip all times after lead_car exists
            
            rel_x.append( smooth_car[idx,1] - smoothed[0][2][lead_idx,1])
            rel_xt.append(smooth_car[idx,0])
            
        ax[1].plot(rel_xt,torch.tensor(rel_x),color = colors[t])
                
        
        # plot speed
        ax[2].plot(smooth_car[:-1,0],-1*(smooth_car[1:,1]-smooth_car[:-1,1])/(smooth_car[1:,0]-smooth_car[:-1,0]),color = colors[t])
        leg.append("Platoon vehicle {} (id {})".format(t,tidx))
    
    
    # Make plot pretty
    ax[2].set_xlabel("Time (s)")
    ax[1].set_ylabel("Distance Behind Lead Vehicle (ft)")
    ax[0].set_ylabel("X Position (ft)")
    ax[2].set_ylabel("Velocity (ft/s)")
    fig.legend(leg)
    ax[0].set_title("Platoon Plot")

    
    
    #%% 7. Plot boxes for a few cameras on video data
    if __name__ == "__main__":
        from data_viewer import VideoViewer
    
        video_dir = "/home/worklab/Documents/datasets/I24-V/video"
        camera_names   = ["P32C02","P32C04","P32C05","P32C06"]
           
        dv = VideoViewer(video_dir,
                        camera_names,
                        rcs,
                        buffer_frames = 100,
                        start_time = 0, 
                        gps = None,
                        manual = None,
                        detections = new_track_path)
        dv.run()
        # d   - toggle detections on /off
        # []  - switch cameras
        # 8 9 - go back/advance a frame
      
        
   