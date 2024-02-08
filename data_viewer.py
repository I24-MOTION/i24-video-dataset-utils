import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import cv2
import csv
import copy
import sys
import string
import cv2 as cv
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import _pickle as pickle 
import time
import scipy

import time
import random


from torchvision.transforms import functional as F
from torchvision.ops import roi_align, nms


import torch.multiprocessing as mp
ctx = mp.get_context('spawn')

# personal modules and packages
from i24_rcs import I24_RCS
from nvc_buffer2 import NVC_Buffer





import pandas as pd
import numpy as np
import torch




# class with attributes
# GPS data
# Manual box data
# Video
# RCS save file
# Relative time


# inputs - time
# camera names (up to 4)
# optional - buffer window 


class DataViewer:
    """ 
    """
    
    def __init__(self,
                 video_dir,
                 view_cameras,
                 hg_path,
                 start_time, 
                 buffer_frames = 60,
                 gps = None,
                 manual = None,
                 detections = None):
        
        """
        video_dir - (str) directory with video sequences
        camera_list - [str] camera names to plot 
        hg_path     - str - path to homography / rcs save file (.cpkl)
        start_time  - float - timestamp at which to start viewing
        buffer_frames  - int - number of frames of video to buffer after start_time
        gps         - str - path to gps data save file
        manual      - str path to manual annotation save file
        detections  - str path to detections save file
        """
        
        
        #### Initialize frame array
        self.active_direction = -1

        # get list of all cameras available
        camera_names = os.listdir(video_dir)
        camera_names.sort()
        camera_names.reverse()
        
        include_cameras = []
        for camera in camera_names:
            if ".pts" in camera: continue
            shortname = camera.split("_")[0]
            if ".mkv" in camera and shortname in view_cameras:
                include_cameras.append(shortname)
                
           
        self.camera_names = include_cameras
        # 1. Get multi-thread frame-loader object
        self.b = NVC_Buffer(video_dir,include_cameras,ctx,buffer_lim = buffer_frames,start_time = start_time)
        
        
       
        
        #### Initialize a bunch of other stuff for tool management
        self.cont = True
        self.colors =  np.random.rand(2000,3)
        self.frame_idx = 0
        self.active_cam = 0
        self.buffer(buffer_frames)
        
        
        #### get homography
        self.hg = I24_RCS(save_path = hg_path,downsample = 2)
        self.hg.hg_start_time = 0
        self.hg.hg_sec = 10
        
        ### get manual data 
        if manual is None:
            self.manual = None
        else:
            self.load_manual(manual)
            
        ### get GPS data
        if gps is None:
            self.gps = None
        else:
            self.load_gps(gps)
        
        ### get detections
        if detections is None:
            self.detections = None
        else:
            self.load_detections(detections)  
       
        # for plotting detections
        self.d1 = len(detections) -1 
        self.d2 = 0
        self.DET = True
    
    
        
    def load_gps(self,gps_path):
        self.gps = pd.read_csv(gps_path)
        
        self.gps.rename(columns={"Width (ft)":"w","Length (ft)":"l","Height (ft)":"h", "Roadway X (ft)":"x","Roadway Y (ft)":"y","Timestamp (s)":"t"},inplace = True)
        del self.gps["State Plane X (ft)"]
        del self.gps["State Plane Y (ft)"]
    
    def load_manual(self,manual_path):
        self.manual = pd.read_csv(manual_path)
        self.manual.rename(columns={"Width (ft)":"w","Length (ft)":"l","Height (ft)":"h", "Roadway X (ft)":"x","Roadway Y (ft)":"y","Timestamp (s)":"t"},inplace = True)
        
    def load_detections(self,detection_path):
        self.detections = np.load(detection_path)
            
    def cache_frame_data(self):
        if len(self.data[self.frame_idx] ) > 0:
            return
        else:
            
            gps_margin = 0.01
            ts = self.buffer.ts[self.frame_idx][self.active_cam]
            frame_gps = self.gps[self.gps['t'].between(ts-gps_margin,ts+gps_margin)]
        
    def quit(self):      
        self.cont = False
        cv2.destroyAllWindows()
            
        self.save()
        
        
    def save(self):
        with open(self.save_file,"wb") as f:
            pickle.dump([self.data,self.objects],f)
        print("Saved annotations at {}".format(self.save_file))
        
        self.recount_objects()
        
    def buffer(self,n):
        self.b.fill(n)
        #while len(self.b.frames[self.frame_idx]) == 0:
        #self.next(stride = n-1)
            
    def safe(self,x):
        """
        Casts single-element tensor as an variable, otherwise does nothing
        """
        try:
            x = x.item()
        except:
            pass
        return x

        
    def toggle_cams(self,dir):
        """dir should be -1 or 1"""
        
        if self.active_cam + dir < len(self.camera_names) -1 and self.active_cam + dir >= 0:
            self.active_cam += dir
            self.plot()
       
      
               
    def next(self,stride = 1):
        """
        We assume all relevant frames are pre-buffered, so all we have to do is 
        Check whether there's another frame available to advance, advance the index,
        and then assign the indexed frame and timestamps
        """        

        if self.frame_idx+stride < len(self.b.frames):
            self.frame_idx += stride
            
        else:
            print("On last frame")
    
    
    
    def prev(self,stride = 1):
        
        
        if self.frame_idx-stride >= 0 and len(self.b.frames[self.frame_idx-stride]) > 0:
            self.frame_idx -= stride     


        else:
            print("Cannot return to previous frame. First frame or buffer limit")
                        
    def plot(self):        
        plot_frames = []
        #ranges = self.ranges
        

        for i in range(self.active_cam, self.active_cam+2):
           frame = self.b.frames[self.frame_idx][i]
           frame = frame.copy()
           frame_ts = self.b.ts[self.frame_idx][i]
           
           if self.DET:
               # plot Detections - np array of:  "time (s)","x_pos (ft)","y_pos (ft)","length (ft)","width (ft)","height (ft)","class","det confidence"
               start = time.time()
               det_margin = 0.02
               
               # update detection slice start
               if self.detections[self.d1,0] > frame_ts - det_margin:
                   while self.detections[self.d1,0] > frame_ts - det_margin:
                       self.d1 -= 1
               elif self.detections[self.d1,0] < frame_ts - det_margin:
                    while self.detections[self.d1,0] < frame_ts - det_margin:
                        self.d1 += 1
               
               if self.detections[self.d2,0] > frame_ts + det_margin:
                    while self.detections[self.d2,0] > frame_ts + det_margin:
                        self.d2 -= 1
               elif self.detections[self.d2,0] < frame_ts + det_margin:
                     while self.detections[self.d2,0] < frame_ts + det_margin:
                         self.d2 += 1
                       
               boxes = torch.from_numpy(self.detections[self.d1:self.d2,[1,2,3,4,5]].astype(float))
               boxes = torch.cat((boxes,torch.sign(boxes[:,1]).unsqueeze(1)),dim = 1)
               if boxes.shape[0] > 0:
                    self.hg.plot_state_boxes(frame,boxes,labels = None,times = frame_ts, name = [self.camera_names[i] for _ in boxes],color = (255,255,255),thickness = 1)
           
           # plot GPS
           gps_margin = 0.06
           frame_gps = self.gps[self.gps['t'].between(frame_ts-gps_margin,frame_ts+gps_margin)]
           ids = frame_gps["id"].to_list()
           frame_gps = frame_gps.to_numpy()
           boxes = torch.from_numpy(frame_gps[:,[1,2,4,5,6]].astype(float))
           boxes = torch.cat((boxes,torch.sign(boxes[:,1]).unsqueeze(1)),dim = 1)
           
           if len(ids) > 0:
               self.hg.plot_state_boxes(frame,boxes,labels = ids,times = frame_ts, name = [self.camera_names[i] for _ in boxes],color = (0,255,0),thickness = 2)
           
           
           # plot Manual
           manual_margin = 0.06
           frame_manual = self.manual[self.manual['t'].between(frame_ts-manual_margin,frame_ts+manual_margin)]
           ids = frame_manual["id"].to_list()
           frame_manual = frame_manual.to_numpy()
           boxes = torch.from_numpy(frame_manual[:,[3,1,4,5,6]].astype(float))
           boxes = torch.cat((boxes,torch.sign(boxes[:,1]).unsqueeze(1)),dim = 1)
           
           if len(ids) > 0:
               itetwe = 1
               self.hg.plot_state_boxes(frame,boxes,labels = ids,times = frame_ts, name = [self.camera_names[i] for _ in boxes],color = (255,255,0),thickness = 4)
           
           
          
          
          
           
           
           if True:
               font =  cv2.FONT_HERSHEY_SIMPLEX
               header_text = "{} frame {}: {:.3f}s".format(self.camera_names[i],self.frame_idx,self.b.ts[self.frame_idx][i])
               frame = cv2.putText(frame,header_text,(30,30),font,1,(255,255,255),1)
               
           plot_frames.append(frame)
       
        # concatenate frames
        n_ims = len(plot_frames)
        n_row = int(np.round(np.sqrt(n_ims)))
        n_col = int(np.ceil(n_ims/n_row))
        
        cat_im = np.zeros([1080*n_row,1920*n_col,3]).astype(float)
        for i in range(len(plot_frames)):
            im = plot_frames[i]
            row = i // n_row
            col = i % n_row
            
            cat_im[col*1080:(col+1)*1080,row*1920:(row+1)*1920,:] = im
            
        # view frame and if necessary write to file
        cat_im /= 255.0
        
        
        self.plot_frame = cat_im
        
   
   
    
    def return_to_first_frame(self):
        #1. return to first frame in buffer
        for i in range(0,len(self.b.frames)):
            if len(self.b.frames[i]) > 0:
                break
            
        self.frame_idx = i

        
    
    def box_to_state(self,point,direction = False):
        """
        Input box is a 2D rectangle in image space. Returns the corresponding 
        start and end locations in space
        point - indexable data type with 4 values (start x/y, end x/y)
        state_point - 2x2 tensor of start and end point in space
        """
        point = point.copy()
        #transform point into state space
        if point[0] > 1920:
            cam = self.camera_names[self.active_cam+1]
            point[0] -= 1920
            point[2] -= 1920
        else:
            cam = self.camera_names[self.active_cam]

        point1 = torch.tensor([point[0],point[1]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        point2 = torch.tensor([point[2],point[3]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        point = torch.cat((point1,point2),dim = 0)
        
        state_point = self.hg.im_to_state(point,name = [cam,cam], heights = torch.tensor([0]))
        
        return state_point[:,:2]
    
        
    
                
    def hop(self):
        self.next(stride = 30)
            
    
        
    # def on_mouse(self,event, x, y, flags, params):
    #    if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
    #      self.start_point = (x,y)
    #      self.clicked = True 
    #    elif event == cv.EVENT_LBUTTONUP:
    #         box = np.array([self.start_point[0],self.start_point[1],x,y])
    #         self.new = box
    #         self.clicked = False
            
            
    #         if x > 1920:
    #             self.clicked_camera = self.camera_names[self.active_cam+1]
    #             self.clicked_idx = self.active_cam + 1
    #         else:
    #             self.clicked_camera = self.camera_names[self.active_cam]
    #             self.clicked_idx = self.active_cam
        
    #    # some commands have right-click-specific toggling
    #    elif event == cv.EVENT_RBUTTONDOWN:
    #         self.right_click = not self.right_click
    #         self.copied_box = None
            
       # elif event == cv.EVENT_MOUSEWHEEL:
       #      print(x,y,flags)
    
   

    def run(self):
        """
        Main processing loop
        """
        
        cv2.namedWindow("window")
        #cv.setMouseCallback("window", self.on_mouse, 0)
        self.plot()
        self.cont = True
        
        while(self.cont): # one frame
        
           ### Show frame
           cv2.imshow("window", self.plot_frame)
           title = "Frame {}, Cameras {} and {}".format(self.frame_idx,self.camera_names[self.active_cam],self.camera_names[self.active_cam + 1])
           cv2.setWindowTitle("window",str(title))
           
           
           ### Handle keystrokes 
           key = cv2.waitKey(1)

           
           if key == ord('9'):
                self.next()
                self.plot()
           elif key == ord("b"):
               self.hop()
               self.plot()
                
           elif key == ord('8'):
                self.prev()  
                self.plot()
                
           elif key == ord('f'):
                self.return_to_first_frame()  
                self.plot()     
           elif key == ord("d"):
               self.DET = not self.DET  
            
           elif key == ord("q"):
               self.quit()
           
           elif key == ord("["):
               self.toggle_cams(-1)
               
           elif key == ord("]"):
               self.toggle_cams(1)
        
           elif key == ord("+"):
               print("Filling buffer. Type number of frames to buffer...")
               n = int(self.keyboard_input())  
               self.buffer(n)
               
               
if __name__ == "__main__":
    
    
    # specify inputs

    gps_path       = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/final_gps.csv"  # path to adjusted GPS data (optional)
    manual_path    = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/final_manual.csv" # path to manually labeled box data (optional)
    detection_path = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/final_detections.npy" # path to detection save file (optional)
    video_dir      = "/home/worklab/Documents/temp_wacv_video" # path to video sequence directory
    hg_path        = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/WACV2024_hg_save.cpkl" # path to hg.cpkl save file
    camera_names   = ["P20C01","P20C02","P20C03","P20C04","P20C05","P20C06"]
    buffer_window  = 4000 # frames load starting with specified time
    start_time     = 200   # timestamp in seconds (first frame is 0 according to timestamps)
    
    
    dv = DataViewer(video_dir,
                    camera_names,
                    hg_path,
                    buffer_frames = buffer_window,
                    start_time = start_time, 
                    gps = gps_path,
                    manual = manual_path,
                    detections = detection_path)
    dv.run()
  
            
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
           