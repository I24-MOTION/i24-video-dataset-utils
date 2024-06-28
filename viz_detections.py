import numpy as np
import pandas as pd
import cv2
import _pickle as pickle
import torch


df   = "/home/worklab/Documents/datasets/I24-V/final_detections.npy"
gpsf = "/home/worklab/Documents/datasets/I24-V/final_gps.csv"



class Viewer:
    def __init__(self,det,gps):
        

        self.load_idx = 0
        self.startx = 0
        self.endx = 15000#4000
        self.starty = -36
        self.endy = -23
        
        self.plot_history = 300
        self.capture = False
        self.frame_idx = 0
        
        

        
        self.grid = True
        self.history = False
        self.grid_major = 200
        self.grid_minor = 25
        
        self.cur_time = 0
        self.cur_time_idx = 0
        
        self.detection_buffer = []
        self.gps_buffer = []
        self.next_det_idx = 0
        self.next_gps_idx = 0
        
        

        
        self.det = det
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
        zvp = 1080,1920
        origin  = 1080,1920

        
        self.P = torch.tensor([[xvp[0]/xs,yvp[0]/ys,zvp[0]/zs,origin[0]],
                          [xvp[1]/xs,yvp[1]/ys,zvp[1]/zs,origin[1]],
                          [xs,ys,zs,1]]).double()
        self.P[:,:3] *= 0.1
        self.clicked = False
        self.active_vp = -1
    
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
            self.starty = param["endy"]
        if "endy" in param.keys():
            self.endy = param["endy"]
        if "grid_major" in param.keys():
            self.grid_major = param["grid_major"]
        
        print("Loaded parameter setting {}".format(self.load_idx))
        
        # cycles through all saves
        self.load_idx = (self.load_idx + 1) % len(params)
        
            
    def next(self):
        
        self.cur_time += 0.1
        self.cur_time_idx += 1
        
        if self.cur_time_idx > len(self.detection_buffer) -1:
            self.get_time_detections()
            self.get_time_gps()
        
        self.plot()
        
    
    def prev(self):
        if self.cur_time_idx > 0:
            self.cur_time -= 0.1
            self.cur_time_idx -= 1
            
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
            if x > self.startx and x < self.endx and y > self.starty and y < self.endy: 
                selection.append(self.det[idx])
            
            idx += 1
            

        
        # save in detection buffer
        self.next_det_idx = idx
        self.detection_buffer.append(selection)
        
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
        if self.history: ph = self.plot_history 
        else: ph = 0
        
        for idx in np.arange(self.cur_time_idx-ph,self.cur_time_idx+1,1):
        
            if idx >= 0:
                
                color = (255,255,255)
                if idx != self.cur_time_idx:
                    color = (200,200,0)
                
                # stack detections
                det = self.detection_buffer[idx]
                det = torch.from_numpy(np.array(det)).double()
                det = det[:,[1,2,3,4,5]]
                
                
                # convert from xylwh to x1y1x2y2
                n_det = det.shape[0]
                x1 = det[:,0]                -self.startx
                x2 = det[:,0] + det[:,2]     -self.startx
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
                for d in new_pts:
                    p1 = int(d[0,1]),int(d[0,0])
                    p2 = int(d[1,1]),int(d[1,0])
                    p3 = int(d[2,1]),int(d[2,0])
                    p4 = int(d[3,1]),int(d[3,0])
                    
                    cv2.line(self.plot_frame,p1,p2,color,1)
                    cv2.line(self.plot_frame,p3,p4,color,1)
                    cv2.line(self.plot_frame,p1,p3,color,1)
                    cv2.line(self.plot_frame,p2,p4,color,1)
                    
       
        if self.capture:
            cv2.imwrite("frames/{}.png".format(str(self.frame_idx).zfill(5)),self.plot_frame)
            self.frame_idx += 1             
       
        
    def plot(self):
        
        if True:
            self.plot3D()
            return
        
        self.plot_frame = np.zeros([200,int(self.ppf*(self.endx-self.startx)),3]).astype(np.uint8)
        
        
        # plot grid
        if self.grid:
            for y in [0,-12,-24,-36,-48,-60,-72,12,24,36,48,60,72]:
                color = (40,40,40)
                if y in [-12,12]:
                    color = (0,255,255)
                y += 100
                cv2.line(self.plot_frame,(0,y),(self.plot_frame.shape[1]-1,y),color,1)
        
            
            for x in np.arange(0,30000,self.grid_minor):
                if x < self.startx or x > self.endx:
                    continue
                color = (40,40,40)
                cv2.line(self.plot_frame,(x-self.startx,0),(x-self.startx,self.plot_frame.shape[0]-1),color,1)
        
            for x in np.arange(0,30000,self.grid_major):
                if x < self.startx or x > self.endx:
                    continue
                color = (100,100,100)
                cv2.line(self.plot_frame,(x-self.startx,0),(x-self.startx,self.plot_frame.shape[0]-1),color,1)
                cv2.putText(self.plot_frame, str("{}ft".format(x)), (x-self.startx,self.plot_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, -0.4, color)                

        
        # plot gps buffer
        for idx in np.arange(self.cur_time_idx-10,self.cur_time_idx+1,1):
        
            if idx >= 0:
                
                det = self.gps_buffer[idx]
                
                color = (0,0,255)
                if idx != self.cur_time_idx:
                    color = (0,0,75)
        
                for d in det:
                    x1 = int(d[1]) - self.startx
                    x2 = int(d[1] + d[3]) - self.startx
                    y1 = int(d[2] - 0.5*d[4]) + 100
                    y2 = int(d[2] + 0.5*d[4]) + 100 
                    
                    cv2.rectangle(self.plot_frame,(x1,y1),(x2,y2),color,-1)


        # plot detection buffer for most recent 3 frames
        
        for idx in np.arange(self.cur_time_idx-4,self.cur_time_idx+1,1):
        
            if idx >= 0:
                
                det = self.detection_buffer[idx]
                
                color = (255,255,255)
                if idx != self.cur_time_idx:
                    color = (50,50,0)
        
                for d in det:
                    x1 = int(d[1]) - self.startx
                    x2 = int(d[1] + d[3]) - self.startx
                    y1 = int(d[2] - 0.5*d[4]) + 100
                    y2 = int(d[2] + 0.5*d[4]) + 100 
                    
                    cv2.rectangle(self.plot_frame,(x1,y1),(x2,y2),color,1)
                

    
        self.plot_frame = cv2.rotate(self.plot_frame, cv2.ROTATE_180)
    
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
                       
           if key == ord('r'):
               # for i in range(3600):
               #     self.next()
               
               self.cur_time += 60
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
                self.history = not self.history
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
               self.adjust_vp(0.8)
               self.plot()          

           elif key == ord("]"):
               self.adjust_vp(1.25)
               self.plot()          

           elif key == ord ("@"):
               self.save()
           elif key == ord ("#"):
               self.load()
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
               for t in range(3000):
                   self.capture = True
                   self.next()
                   if t % 100 == 0:
                       print("On frame {}".format(t))
                   self.capture = False
            
                
if __name__ == "__main__":

    # open gps file
    det = np.load(df)
    
    # open df
    gps = pd.read_csv(gpsf)
    
    
    
    v = Viewer(det,gps)
    v.run()
    
