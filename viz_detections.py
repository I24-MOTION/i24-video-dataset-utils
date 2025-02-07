import numpy as np
import pandas as pd
import cv2
import _pickle as pickle
import torch


df   = "/home/worklab/Documents/datasets/I24-V/final_detections.npy"
df   = "data/kiou_detections.npy"
df = "/home/worklab/Documents/datasets/2025-2-gamma-tracking/MOTION_NEW_TIME.npy"
#df = "/home/worklab/Documents/i24/i24-video-dataset-utils/CHANGEME_OLD_DATA.npy"
gf   = "data/gap_detections.npy"
#df = "data/clustered_0_100.npy"
gpsf = "/home/worklab/Documents/datasets/I24-V/final_gps.csv"



class Viewer:
    def __init__(self,det,gps,gap = None):
        

        self.load_idx = 0
        self.startx = 3000
        self.endx = 10000#4000
        self.starty = -60
        self.endy = 0
        
        self.plot_histories = [0,3,10,30,100,200,4000]
        self.capture = False
        self.frame_idx = 0
        self.multicolor = True
        self.dot_history = False

        
        self.grid = True
        self.history = 0
        self.grid_major = 200
        self.grid_minor = 25
        
        self.cur_time = 0
        self.cur_time_idx = 0
        
        self.detection_buffer = []
        self.gps_buffer = []
        self.next_det_idx = 0
        self.next_gps_idx = 0
        self.det = det
        
        self.gap = None
        if gap is not None:
            self.gap = gap
            self.gap_buffer = []
            self.next_gap_idx = 0
            self.get_time_gaps()
            
        #self.det = det[det[:, 0].sort()]
        #self.det[self.det[:, 0].argsort()]
        
        self.gps = gps.sort_values(by=['Timestamp (s)']).to_numpy()
        
        self.get_time_detections()
        self.get_time_gps()
        
        
        self.ppf = 1
        
        # camera model params
        # self.rx = np.pi/4
        # self.ry = np.pi/4
        # self.rz = np.pi/4
        
        # self.tx = 100
        # self.ty = 100
        # self.tz = 1000
        # self.f  = 100
        
        # initialize P
        ys = 1
        xs = -4
        zs = -1
        xvp = 0,3840,0
        yvp = 0,0,0
        zvp = 1080*2,1920*2
        origin  = 1080*2,1920*2

        
        self.P = torch.tensor([[xvp[0]/xs,yvp[0]/ys,zvp[0]/zs,origin[0]],
                          [xvp[1]/xs,yvp[1]/ys,zvp[1]/zs,origin[1]],
                          [xs,ys,zs,1]]).double()
        self.P[:,:3] *= 0.1
        self.clicked = False
        self.active_vp = -1
        
        self.color = np.random.randint(0,255,[1000,3]).astype(np.uint8)
        self.color[1] = [255,255,0]
        self.color[2] = [255,0,255]
        self.color[3] = [0,255,255]
        self.color[4] = [220,255,255]
    
    def save(self):
        
        try:
            with open("viewer.param", "rb") as f:
                params = pickle.load(f)
        except FileNotFoundError:
            params = []
            
        new_params = {
            "P":self.P,
            "startx":self.startx,
            "endx":self.endx,
            "starty":self.starty,
            "endy":self.endy,
            "grid_major":self.grid_major,
            }
        
        params.append(new_params)
            
        with open("viewer.param", "wb") as f:
            pickle.dump(params,f)
            
        print("Saved current parameters")
        
        
    def load(self):
        
        with open("viewer.param", "rb") as f:
            params = pickle.load(f)
        
        param = params[self.load_idx]
        
        if "P" in param.keys():
            self.P = param["P"]
        if "startx" in param.keys():
            self.startx = param["startx"]
        if "endx" in param.keys():
            self.endx = param["endx"]
        if "starty" in param.keys():
            self.starty = param["starty"]
        if "endy" in param.keys():
            self.endy = param["endy"]
        if "grid_major" in param.keys():
            self.grid_major = param["grid_major"]
        
        print("Loaded parameter setting {}".format(self.load_idx))
        
        # cycles through all saves
        self.load_idx = (self.load_idx + 1) % len(params)
        
    def remove_preset(self):
        with open("viewer.param", "rb") as f:
            params = pickle.load(f)
            
            if self.load_idx > 0:
                
                del params[self.load_idx - 1]
            
            with open("viewer.param", "wb") as f:
                pickle.dump(params,f)
                
            print("Deleted preset {}".format(self.load_idx))
            
            self.load_idx -= 1
            self.load()
    
    def next(self):
        
        self.cur_time += 0.1
        self.cur_time_idx += 1
        
        if self.cur_time_idx > len(self.detection_buffer) -1:
            self.get_time_detections()
            self.get_time_gps()
            
            if self.gap is not None:
                self.get_time_gaps()
        
        self.plot()
        
    
    def prev(self):
        if self.cur_time_idx > 0:
            self.cur_time -= 0.1
            self.cur_time_idx -= 1
            
            if self.gap is not None:
                self.get_time_gaps()
            
            self.plot()
            
        else:
            print("At start of sequence, cannot move backwards in time")

    
    def get_time_detections(self):
        selection = []
        idx = self.next_det_idx
        
        # starting at next_det_idx, advance until time is out of 0.1s window
        while self.det[idx,0] < self.cur_time + 0.1:
            #grab all detections within this range 
            
            # filter by x range
            x =  self.det[idx,1]  
            y = self.det[idx,2]
            if x > self.startx and x < self.endx:# and y > self.starty and y < self.endy: 
                selection.append(self.det[idx])
            
            idx += 1
            
        # save in detection buffer
        self.next_det_idx = idx
        self.detection_buffer.append(selection)
            
    def get_time_gaps(self):
        selection = []
        idx = self.next_gap_idx
        
        # starting at next_det_idx, advance until time is out of 0.1s window
        while self.gap[idx,0] < self.cur_time + 0.1:
            #grab all detections within this range 
            
            # filter by x range
            x =  self.gap[idx,1]  
            y = self.gap[idx,2]
            if x > self.startx and x < self.endx:# and y > self.starty and y < self.endy: 
                selection.append(self.gap[idx])
            
            idx += 1
        
        # save in detection buffer
        self.next_gap_idx = idx
        self.gap_buffer.append(selection)
        
    def get_time_gps(self):
       selection = []
       idx = self.next_gps_idx
       
       # starting at next_det_idx, advance until time is out of 0.1s window
       while self.gps[idx,5] < self.cur_time + 0.1:
           #grab all gps within this range 
           
           # filter by x range
           x =  self.gps[idx,3]  
           if  x > self.startx and x < self.endx:
               datum = np.array([self.gps[idx,5],
                                self.gps[idx,3],
                                self.gps[idx,4],
                                self.gps[idx,6],
                                self.gps[idx,7],
                                self.gps[idx,8]])           
               selection.append(datum)
           idx += 1
           

       
       # save in detection buffer
       self.next_gps_idx = idx
       self.gps_buffer.append(selection)
        
    def plot3D(self):
        #self.plot_frame = np.zeros([1080,1920,3]).astype(np.uint8)
        self.plot_frame = np.zeros([1080,1920,3]).astype(np.uint8)
        
        P = self.P
        
        # grid
        if self.grid:
            
            
            # plot lane lines
            length = self.endx - self.startx
            grid = [[-1,0,-12,length,0,0],
                    [-1,0,12,length,0,0],
                    [-1,0,24,length,0,0],
                    [-1,0,36,length,0,0],
                    [-1,0,48,length,0,0],
                    [-1,0,60,length,0,0],
                    [-1,0,-24,length,0,0],
                    [-1,0,-36,length,0,0],
                    [-1,0,-48,length,0,0],
                    [-1,0,-60,length,0,0],
                    ]
            det = grid
            det = torch.from_numpy(np.array(det)).double()
            det = det[:,[1,2,3,4,5]]
            
            
            # convert from xylwh to x1y1x2y2
            n_det = det.shape[0]
            x1 = det[:,0] 
            x2 = det[:,0] + det[:,2]
            y1 = det[:,1] 
            
            # [n_detections,2,2]
            #z = torch.zeros(n_det) + self.cur_time_idx  - idx
            z = det[:,4]
            ones = torch.ones(n_det)
            det = torch.stack(([x1,y1,z,ones,
                               x2,y1,z,ones,
                               ]),dim = 1).view(n_det,2,4)
            
            
            
            # Project grid
            points = det.view(-1,4)                 # alias
            P2 = P.unsqueeze(1).repeat(1,points.shape[0],1,1).reshape(-1,3,4)
            points = points.unsqueeze(1).transpose(1,2)
            new_pts = torch.bmm(P2,points).squeeze(2)
          
            
            # divide each point 0th and 1st column by the 2nd column
            new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
            new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
            
            # drop scale factor column
            new_pts = new_pts[:,:2] 
            
            # reshape to [d,m,2]
            new_pts = new_pts.view(n_det,-1,2)
            
            for gidx,d in enumerate(new_pts):
                p1 = int(d[0,1]),int(d[0,0])
                p2 = int(d[1,1]),int(d[1,0])
                
                color = (100,100,100)
                if gidx in [0,1]:
                    color = (0,255,255)
                cv2.line(self.plot_frame,p1,p2,color,1)
             
                
             
                
            # plot perp lines
            g = self.grid_major
            xstart = ((self.startx // g) + 1) * g
            xend = xm = (self.endx // g) * g
            grid = torch.arange(xstart,xend,g)
            
            zer = torch.zeros(grid.shape[0])
            w   = torch.ones(grid.shape[0]) * 120
            
            grid = torch.stack([zer,grid,zer,zer,w,zer]).transpose(1,0)
            
            det = grid
            det = torch.from_numpy(np.array(det)).double()
            det = det[:,[1,2,3,4,5]]
            
            
            # convert from xylwh to x1y1x2y2
            n_det = det.shape[0]
            x1 = det[:,0] -self.startx
            
            y1 = det[:,1] + det[:,3]/2
            y2 = det[:,1] - det[:,3]/2.0
            
            # [n_detections,2,2]
            #z = torch.zeros(n_det) + self.cur_time_idx  - idx
            z = det[:,4]
            ones = torch.ones(n_det)
            det = torch.stack(([x1,y1,z,ones,
                               x1,y2,z,ones,
                               ]),dim = 1).view(n_det,2,4)
            
            
            
            # Project grid
            points = det.view(-1,4)                 # alias
            P2 = P.unsqueeze(1).repeat(1,points.shape[0],1,1).reshape(-1,3,4)
            points = points.unsqueeze(1).transpose(1,2)
            new_pts = torch.bmm(P2,points).squeeze(2)
          
            
            # divide each point 0th and 1st column by the 2nd column
            new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
            new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
            
            # drop scale factor column
            new_pts = new_pts[:,:2] 
            
            # reshape to [d,m,2]
            new_pts = new_pts.view(n_det,-1,2)
            
            for gidx,d in enumerate(new_pts):
                p1 = int(d[0,1]),int(d[0,0])
                p2 = int(d[1,1]),int(d[1,0])
                
                color = (100,100,100)
                cv2.line(self.plot_frame,p1,p2,color,1) 
                cv2.putText(self.plot_frame, str("{}ft".format(grid[gidx,1].item())), (p1[0] - 30,p1[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)                

                    
                
        #### Finally, plot detections
        ph = self.plot_histories[self.history]
        if True:
            for idx in np.arange(self.cur_time_idx-ph,self.cur_time_idx+1,1):
            
                if idx >= 0:
                    
                    color = (255,255,255)
                    # if idx != self.cur_time_idx:
                    #     color = (100,100,100)
                        
                       
                            
                    
                    # stack detections
                    det = self.detection_buffer[idx]
                    if len(det) == 0:
                        continue
                    det = torch.from_numpy(np.array(det)).double()
                    
                    
                    keep = (torch.where(det[:,2] > self.starty,1,0) * torch.where(det[:,2] < self.endy,1,0)).nonzero().squeeze()
                    det = det[keep]
                    
                    if len(det) == 0:
                        continue
                    
                    color_idxs = None
                    if det.shape[1] in  [9,8]:
                        color_idxs = det[:,-1].int()
                    else:
                        color_idxs = det[:,2].int() // -12
                        
                        
                        
                    det = det[:,[1,2,3,4,5]]
                    
                    if self.dot_history and idx != self.cur_time_idx:
                        det[:,2:4] = 1
                    
                    # convert from xylwh to x1y1x2y2
                    n_det = det.shape[0]
                    x1 = det[:,0]                                     -self.startx
                    x2 = det[:,0] + det[:,2]*torch.sign(det[:,1])     -self.startx
                    y1 = det[:,1] - det[:,3]/2.0
                    y2 = det[:,1] + det[:,3]/2.0
                    
                    # [n_detections,2,2]
                    z = torch.zeros(n_det) - (self.cur_time_idx  - idx)
                    #z = det[:,4]
                    ones = torch.ones(n_det)
                    det = torch.stack(([x1,y1,z,ones,
                                        x2,y1,z,ones,
                                        x1,y2,z,ones,
                                        x2,y2,z,ones]),dim = 1).view(n_det,4,4)
                    
                    
                    
                    # Project detections
                    points = det.view(-1,4)                 # alias
                    P2 = P.unsqueeze(1).repeat(1,points.shape[0],1,1).reshape(-1,3,4)
                    points = points.unsqueeze(1).transpose(1,2)
                    new_pts = torch.bmm(P2,points).squeeze(2)
                  
                    
                    # divide each point 0th and 1st column by the 2nd column
                    new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                    new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                    
                    # drop scale factor column
                    new_pts = new_pts[:,:2] 
                    
                    # reshape to [d,m,2]
                    new_pts = new_pts.view(n_det,-1,2)
                    
                    
            
                    # plot each detection as lines
                    
                    for didx,d in enumerate(new_pts):
                        p1 = int(d[0,1]),int(d[0,0])
                        p2 = int(d[1,1]),int(d[1,0])
                        p3 = int(d[2,1]),int(d[2,0])
                        p4 = int(d[3,1]),int(d[3,0])
                        
                        if color_idxs is not None:# and idx == self.cur_time_idx:# and self.multicolor == True:
                            color = self.color[color_idxs[didx]%1000].astype(np.uint8)
                            color = int(color[0]),int(color[1]),int(color[2])
                        
                            
                        cv2.line(self.plot_frame,p1,p2,color,1)
                        cv2.line(self.plot_frame,p3,p4,color,1)
                        cv2.line(self.plot_frame,p1,p3,color,1)
                        cv2.line(self.plot_frame,p2,p4,color,1)
                        
                    
                    
        #### Finally, plot gaps
        if self.gap is not None:
            for idx in np.arange(self.cur_time_idx-ph,self.cur_time_idx+1,1):
                 
                     if idx >= 0:
                         
                         color = (255,255,255)
                        
                             
                            
                                 
                         
                         # stack detections
                         det = self.gap_buffer[idx]
                         if len(det) == 0:
                             continue
                         det = torch.from_numpy(np.array(det)).double()
                         
                         
                         keep = (torch.where(det[:,2] > self.starty,1,0) * torch.where(det[:,2] < self.endy,1,0)).nonzero().squeeze()
                         det = det[keep]
                         
                         if len(det) == 0:
                             continue
                         
                         # color_idxs = None
                         # if det.shape[1] == 9:
                         #     color_idxs = det[:,-1].int()
                         # else:
                         #     color_idxs = det[:,2].int() // -12
                             
                             
                             
                         det = det[:,[1,2,3,4,5]]
                         
                         # if self.dot_history and idx != self.cur_time_idx:
                         #     det[:,2:4] = 1
                         
                         # convert from xylwh to x1y1x2y2
                         n_det = det.shape[0]
                         x1 = det[:,0]                                    -self.startx
                         x2 = det[:,0] + det[:,2]*torch.sign(det[:,1])    -self.startx
                         y1 = det[:,1] - det[:,3]/2.0
                         y2 = det[:,1] + det[:,3]/2.0
                         
                         # [n_detections,2,2]
                         z = torch.zeros(n_det) - (self.cur_time_idx  - idx)
                         #z = det[:,4]
                         ones = torch.ones(n_det)
                         det = torch.stack(([x1,y1,z,ones,
                                             x2,y1,z,ones,
                                             x1,y2,z,ones,
                                             x2,y2,z,ones]),dim = 1).view(n_det,4,4)
                         
                         
                         
                         # Project detections
                         points = det.view(-1,4)                 # alias
                         P2 = P.unsqueeze(1).repeat(1,points.shape[0],1,1).reshape(-1,3,4)
                         points = points.unsqueeze(1).transpose(1,2)
                         new_pts = torch.bmm(P2,points).squeeze(2)
                       
                         
                         # divide each point 0th and 1st column by the 2nd column
                         new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                         new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                         
                         # drop scale factor column
                         new_pts = new_pts[:,:2] 
                         
                         # reshape to [d,m,2]
                         new_pts = new_pts.view(n_det,-1,2)
                         
                         
                 
                         # plot each detection as lines
                         
                         for didx,d in enumerate(new_pts):
                             p1 = int(d[0,1]),int(d[0,0])
                             p2 = int(d[1,1]),int(d[1,0])
                             p3 = int(d[2,1]),int(d[2,0])
                             p4 = int(d[3,1]),int(d[3,0])
                             
                             # if color_idxs is not None:# and self.multicolor == True:
                             #     color = self.color[color_idxs[didx]%1000].astype(np.uint8)
                             #     color = int(color[0]),int(color[1]),int(color[2])
                             
                                 
                             cv2.line(self.plot_frame,p1,p2,color,1)
                             cv2.line(self.plot_frame,p3,p4,color,1)
                             cv2.line(self.plot_frame,p1,p3,color,1)
                             cv2.line(self.plot_frame,p2,p4,color,1)   
                    
       
        if self.capture:
            cv2.imwrite("frames/{}.png".format(str(self.frame_idx).zfill(5)),self.plot_frame)
            self.frame_idx += 1             
       
        
    def plot(self):       
            self.plot3D()
            return
        
    #     self.plot_frame = np.zeros([200,int(self.ppf*(self.endx-self.startx)),3]).astype(np.uint8)
        
        
    #     # plot grid
    #     if self.grid:
    #         for y in [0,-12,-24,-36,-48,-60,-72,12,24,36,48,60,72]:
    #             color = (40,40,40)
    #             if y in [-12,12]:
    #                 color = (0,255,255)
    #             y += 100
    #             cv2.line(self.plot_frame,(0,y),(self.plot_frame.shape[1]-1,y),color,1)
        
            
    #         for x in np.arange(0,30000,self.grid_minor):
    #             if x < self.startx or x > self.endx:
    #                 continue
    #             color = (40,40,40)
    #             cv2.line(self.plot_frame,(x-self.startx,0),(x-self.startx,self.plot_frame.shape[0]-1),color,1)
        
    #         for x in np.arange(0,30000,self.grid_major):
    #             if x < self.startx or x > self.endx:
    #                 continue
    #             color = (100,100,100)
    #             cv2.line(self.plot_frame,(x-self.startx,0),(x-self.startx,self.plot_frame.shape[0]-1),color,1)
    #             cv2.putText(self.plot_frame, str("{}ft".format(x)), (x-self.startx,self.plot_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, -0.4, color)                

        
    #     # plot gps buffer
    #     for idx in np.arange(self.cur_time_idx-10,self.cur_time_idx+1,1):
        
    #         if idx >= 0:
                
    #             det = self.gps_buffer[idx]
                
    #             color = (0,0,255)
    #             if idx != self.cur_time_idx:
    #                 color = (0,0,75)
        
    #             for d in det:
    #                 x1 = int(d[1]) - self.startx
    #                 x2 = int(d[1] + d[3]) - self.startx
    #                 y1 = int(d[2] - 0.5*d[4]) + 100
    #                 y2 = int(d[2] + 0.5*d[4]) + 100 
                    
    #                 cv2.rectangle(self.plot_frame,(x1,y1),(x2,y2),color,-1)


    #     # plot detection buffer for most recent 3 frames
        
    #     for idx in np.arange(self.cur_time_idx-4,self.cur_time_idx+1,1):
        
    #         if idx >= 0:
                
    #             det = self.detection_buffer[idx]
                
    #             color = (255,255,255)
    #             if idx != self.cur_time_idx:
    #                 color = (50,50,0)
        
    #             for d in det:
    #                 x1 = int(d[1]) - self.startx
    #                 x2 = int(d[1] + d[3]) - self.startx
    #                 y1 = int(d[2] - 0.5*d[4]) + 100
    #                 y2 = int(d[2] + 0.5*d[4]) + 100 
                    
    #                 cv2.rectangle(self.plot_frame,(x1,y1),(x2,y2),color,1)
                

    
    #     self.plot_frame = cv2.rotate(self.plot_frame, cv2.ROTATE_180)
    
    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and not self.clicked:
          self.start_point = (x,y)
          self.clicked = True 
        elif event == cv2.EVENT_LBUTTONUP and self.clicked:
             xtranslation = x - self.start_point[0]
             ytranslation = y - self.start_point[1]
             
             
             if self.active_vp == -1:
                 self.P[1,3] += xtranslation
                 self.P[0,3] += ytranslation
                 self.clicked = False
                 self.plot()
                 
             else:
                self.P[1,self.active_vp] += xtranslation *self.P[2,self.active_vp]
                self.P[0,self.active_vp] += ytranslation * self.P[2,self.active_vp]
                self.clicked = False
                self.plot()
             
             
        # some commands have right-click-specific toggling
        elif event == cv2.EVENT_RBUTTONDOWN:
              self.right_click = not self.right_click
              self.copied_box = None
             
        elif event == cv2.EVENT_MOUSEWHEEL:
              print(x,y,flags)
         
    def adjust_vp(self,f):
        """f = adjustment factor"""
        
        i = self.active_vp
        xtr = self.P[0,i]/self.P[2,i] - self.P[0,3]/self.P[2,3]
        ytr = self.P[1,i]/self.P[2,i] - self.P[1,3]/self.P[2,3]
        
        
        print("vp: {} {}".format(self.P[0,i]/self.P[2,i],self.P[1,i]/self.P[2,i]))
        print("origin: {} {}".format(self.P[0,3]/self.P[2,3],self.P[1,3]/self.P[2,3]))
        print("translation: {} {}".format(xtr,ytr))
        self.P[0,i] = (xtr*f + self.P[0,3]/self.P[2,3]) * self.P[2,i]
        self.P[1,i] = (ytr*f + self.P[1,3]/self.P[2,3]) * self.P[2,i]
    
    def run(self):
        """
        Main processing loop
        """
        
        cv2.namedWindow("window")
        cv2.setMouseCallback("window", self.on_mouse, 0)
        self.plot()
        self.cont = True
        
        while(self.cont): # one frame
        
           ### Show frame
           cv2.imshow("window", self.plot_frame)
           title = "Time {:.3f}s, Space [{},{}]".format(self.cur_time,self.startx,self.endx)#"Frame {}, Cameras {} and {}".format(self.frame_idx,self.camera_names[self.active_cam],self.camera_names[self.active_cam + 1])
           cv2.setWindowTitle("window",str(title))
           
           
           ### Handle keystrokes 
           key = cv2.waitKey(1)

           
           ###############  frame commands  ###########################3333
           if key == ord('w'):
                self.next()
                self.plot()
                
           if key == ord('e'):
                 for _ in range (5): self.next()
                 self.plot()
           if key == ord('t'):
                  for _ in range (2000): self.next()
                  self.plot()
                       
           if key == ord('r'):
               # for i in range(3600):
               #     self.next()
               
               self.cur_time += 2600
               while self.gps[self.next_gps_idx,5] < self.cur_time:
                   self.next_gps_idx += 1
               while self.det[self.next_det_idx,0] < self.cur_time:
                   self.next_det_idx += 1
                   
               self.next()
               self.plot()

           elif key == ord('q'):
                self.prev()  
                self.plot()
            
           elif key == ord("/"):
               cv2.destroyAllWindows()
               break
           
           
           ########### Roadway commands ########################3
       
               
           ################## Plot options ################3
           elif key == ord("g"):
               self.grid = not self.grid
               self.plot()
           elif key == ord("h"):
                self.history = (self.history+1) % len(self.plot_histories)
                self.plot()
               
           elif key == ord("4"):
                  self.P[:,1] /= 1.1
                  self.plot()          
           elif key == ord("5"):
                 self.P[:,1] *= 1.1
                 self.plot()          
                 
           elif key == ord("7"):
                  self.P[:,0] /= 1.1
                  self.plot()          
           elif key == ord("8"):
                 self.P[:,0] *= 1.1
                 self.plot()  
           elif key == ord("1"):
                   self.P[:,2] /= 1.1
                   self.plot()          
           elif key == ord("2"):
                  self.P[:,2] *= 1.1
                  self.plot()  
            
           elif key == ord("-"):
               self.P[:,:3] /= 1.1
               self.plot()          
           elif key == ord("+"):
               self.P[:,:3] *= 1.1
               self.plot()          
               
           elif key == ord("3"):
               self.active_vp = 2
           elif key == ord("6"):
               self.active_vp = 1
           elif key == ord("9"):
               self.active_vp = 0
           elif key == ord("0"):
               self.active_vp = -1
               
           elif key == ord("["):
                if self.endy - self.starty >= 22:
                    self.starty += 12
                else:
                    self.endy += 12 
                    
                self.next()
                self.plot()

           elif key == ord("]"):
               if self.endy - self.starty >= 22:
                   self.endy -= 12
               else:
                   self.starty -= 12 
                   
               self.plot()
               self.next()

           elif key == ord ("@"):
               self.save()
           elif key == ord ("#"):
               self.load()
               self.plot()
           elif key == ord ("$"):
                self.remove_preset()
                self.plot()
                  
           elif key == ord(","):
               self.startx -=20
               self.endx -= 20
               self.next()
               self.plot()
           elif key == ord("."):
               self.startx +=20
               self.endx += 20
               self.next()
               self.plot()
        
           elif key == ord("*"):
               for t in range(300):
                   self.capture = True
                   self.next()
                   if t % 100 == 0:
                       print("On frame {}".format(t))
                   self.capture = False
           elif key == ord("c"):
               self.capture = not self.capture
                
                
if __name__ == "__main__":


    
    if False:
        track = "/home/worklab/Documents/i24/fast-trajectory-annotator/tracking/results_KIOU_10Hz.cpkl"
        with open (track, "rb") as f:
            t = pickle.load(f)
            
        start = t[0].item() # start time
        all_det = [torch.cat((it[0],torch.tensor(idx).unsqueeze(0).unsqueeze(1).expand(it[0].shape[0],1)),dim = 1) for idx,it in enumerate( t[1])]
        all_det = torch.cat(all_det,dim = 0)
        all_det[:,0] -= start
        all_det = all_det[all_det[:, 0].sort()[1]]
        all_det = all_det.data.numpy()
        np.save("kiou_detections.npy",all_det)     
  
    if False: #create gaps as detections
        det = np.load(df)
        
        tmin = 0
        tmax = 3599
        
        idx1 = -1
        idx2 = 0
        
        gap_data = torch.empty([0,7])
        granularity = 1
        
        # iterate through time at 10 Hz 
        for ttime in np.arange(tmin,tmax-1,0.1):
            print("\rParsing time {:.1f} of {:.1f}".format(ttime,tmax),end = "\r", flush = True)
        
            # set idx1 to previous idx2
            idx1 = idx2
            
            # advance idx2 until indexed time is greater than ttime + 0.1
            while det[idx2,0]  < ttime + 0.1:
                idx2 += 1
        
            
            # for each detection, add to raster vector
            objs = det[idx1:idx2,:]
            # 6 inch x 12 lane raster grid for 22000 feet
            raster  = torch.zeros(int(22000/granularity),12)
            
            # for each time, grab detections in each lane
            for d in objs:
                lane_idx = int(d[2]//12 + 6)
                if lane_idx > 11 or lane_idx < 0: continue
                xmin = min(d[1],d[1] + np.sign(d[2])*d[3])
                xmax = max(d[1],d[1] + np.sign(d[2])*d[3])
                minbin = int(xmin/granularity)
                maxbin = int(xmax/granularity)
                
                raster[minbin:maxbin,lane_idx] = 1
                
            if False:
                for lidx in range(12):
                    gap = True
                    gap_start = 0
                    
                    lane_raster = raster[:,lidx]
                    
                    for i in range(len(lane_raster)):
                        if lane_raster[i] == 1 and gap:
                            gap = False
                            gap_end = i* granularity
                            
                            x = gap_end if lidx < 6 else gap_start
                            length = np.abs(gap_end - gap_start)
                            y = (lidx-6)*12 + 6
                            
                            data = torch.tensor([ttime,x,y,length,0,0,-1]).unsqueeze(0)
                            gap_data = torch.cat((gap_data,data),dim = 0)
                            
                        elif lane_raster[i] == 0 and not gap:
                            gap_start = i* granularity
                            gap = True
                            
            if True: 
                raster = 1-raster
                raster = torch.cat((torch.zeros(1,raster.shape[1]),raster,torch.zeros(1,raster.shape[1])),dim = 0)                
                
                
                # crudely ignoring the lost row from finite difference here, would need to properly address later
                raster_shifted = raster[1:,:] - raster[:-1,:] 
                # raster_shifted[0,:] = -1
                # raster_shifted[-1,:] = 1
                                           
                # anywhere where there is a 1, a vehicle starts
                # anywhere where there is a -1, a vehicle ends
                raster_shifted = raster_shifted.transpose(1,0)
                starts = torch.where(raster_shifted == 1,1,0).nonzero()
                ends = torch.where(raster_shifted == -1,1,0).nonzero()
                assert starts.shape == ends.shape
            
                # ends = ends[1:]
                # starts = starts[:-1]
            
                xmin = starts[:,1] * granularity
                xmax = ends[:,1] * granularity
                y_mid = (starts[:,0] - 6) * 12  + 6 # middle of lane
            
                length = torch.abs(xmin-xmax)
                
                direction = torch.where(y_mid > 0,1,0) 
                x = xmin*direction + xmax*(1-direction)
                
                data = torch.stack([torch.zeros(x.shape)+ttime,
                                    x,
                                    y_mid,
                                    length,
                                    torch.zeros(x.shape),
                                    torch.zeros(x.shape),
                                    torch.zeros(x.shape)-1],dim = 1)
                
                gap_data = torch.cat((gap_data,data),dim = 0)
            
            if ttime > 200:
                break
            
        gap_data = gap_data.data.numpy()
        np.save("gap_detections.npy",gap_data)
        
           
        
    # open gps file
    det = np.load(df)
    
    # open df
    gps = pd.read_csv(gpsf)
         
    # open gps file
    gaps = None #np.load(gf)
    
    v = Viewer(det,gps,gap=gaps)
    v.run()
    
