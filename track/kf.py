#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:49:38 2020
@author: worklab
"""

import torch
import numpy as np
import time


class Torch_KF(object):
    """
    A tensor-based Kalman Filter that evaluates many KF-tracked objects in parallel
    using tensor operations
    """
    
    def __init__(self,device,state_err = 10000, meas_err = 1, mod_err = 1, INIT = None, ADD_MEAN_Q = False, ADD_MEAN_R = False):
        """
        Parameters
        ----------
        device : torch.device
            Specifies whether tensors should be stored on GPU or CPU.
        state_err : float, optional
            Specifies the starting state covariance value along the main diagonal. The default is 1.
        meas_err : float, optional
            Specifies the measurement covariance value along the main diagonal. The default is 1.
        mod_err : float, optional
            specifies the model covariance value along the main diagonal. The default is 1.
        INIT : dictionary, optional
            A dictionary containing initialization matrices for P0, H, mu_H, Q, and mu_Q. 
            If specified, these are used instead of the diagonal values
        """
        # initialize tensors
        self.meas_size = 5
        self.state_size = 6
        self.dt_default = 1/30.0
        self.device = device
        self.X = None
        self.D = None
        self.T = None
        
        self.P0 = torch.zeros(self.state_size,self.state_size) # state covariance
        self.F = torch.zeros(self.state_size,self.state_size) # dynamical model
        self.H = torch.zeros(self.meas_size,self.state_size)  # measurement model
        self.Q = torch.zeros(self.state_size,self.state_size) # model covariance
        self.R = torch.zeros(self.meas_size,self.meas_size)   # measurement covariance
        self.R2 = torch.zeros(self.meas_size,self.meas_size)   # second measurement covariance
        
        # obj_ids[a] stores index in X along dim 0 where object a is stored
        self.obj_idxs = {}
        
        if INIT is None:
            # set intial value for state covariance
            self.P0 = torch.eye(self.state_size).unsqueeze(0) * state_err
            
            # these values won't change 
            self.F    = torch.eye(self.state_size).float()
            #self.F[[0,1,2],[4,5,6]] = self.t
            self.H[:4,:4] = torch.eye(4)
            self.Q    = torch.eye(self.state_size).unsqueeze(0) * mod_err                     #+ 1
            self.R    = torch.eye(self.meas_size).unsqueeze(0) * meas_err
            self.R2   = torch.eye(self.meas_size).unsqueeze(0) * meas_err
            self.mu_Q = torch.zeros([1,self.state_size])
            self.mu_R = torch.zeros([1,self.meas_size])

            
        # use INIT matrices to initialize filter    
        else:
            self.P0 = INIT["P"].unsqueeze(0) 
            self.F  = INIT["F"]
            self.H  = INIT["H"]
            
            self.Q  = INIT["Q"]
            if self.Q.dim() == 2: self.Q = self.Q.unsqueeze(0)
            
            self.R  = INIT["R"].unsqueeze(0) 
            self.mu_Q = INIT["mu_Q"].unsqueeze(0) 
            self.mu_R = INIT["mu_R"].unsqueeze(0)
            
            if "R2" in INIT.keys():
                self.R2  = INIT["R2"].unsqueeze(0).to(device).float()  
                self.mu_R2 = INIT["mu_R2"].unsqueeze(0).to(device).float()
                self.H2 = INIT["H2"].to(device).float()
            if "R3" in INIT.keys():
                self.R3  = INIT["R3"].unsqueeze(0).to(device).float()
                self.mu_R3 = INIT["mu_R3"].unsqueeze(0).to(device).float()
                self.H3 = INIT["H3"].to(device).float()
            if "mu_v" in INIT.keys():
                self.mu_v = INIT["mu_v"]
            if "class_size" in INIT.keys():
                self.class_size = INIT["class_size"]
            if "class_covariance" in INIT.keys():
                self.class_covariance = INIT["class_covariance"]
                
            self.state_size = self.F.shape[0]
            self.meas_size  =  self.H.shape[0]
            
            #overwrite means
            if not ADD_MEAN_Q:
                self.mu_Q  = torch.zeros([1,self.state_size])
            if not ADD_MEAN_R:
                self.mu_R  = torch.zeros([1,self.meas_size])
                #self.mu_R2 = torch.zeros([1,self.meas_size])

       
        # move to device
        self.F = self.F.to(device).float()
        self.H = self.H.to(device).float()
        self.Q = self.Q.to(device).float()
        self.R = self.R.to(device).float()
        #self.R2 = self.R.to(device).float()
        self.P0 = self.P0.to(device).float() 
        self.mu_Q = self.mu_Q.to(device).float()
        self.mu_R = self.mu_R.to(device).float()
        #self.mu_R2 = self.mu_R.to(device).float()
    
        # self.y_clamp = torch.tensor([[-5,5],
        #                              [-2,2],
        #                              [-3,3],
        #                              [-2,2],
        #                              [-1,1],
        #                             ]).to(device).float()
        
        self.y_clamp = torch.tensor([[-7,7],
                                     [-1,1],
                                     [-3,3],
                                     [-1,1],
                                     [-1,1],
                                    ]).to(device).float()
    
    def get_dt(self, target_time, idxs = None, use_default = True):
        """
        Given a time, computes dt for each object in the filter. There are 4 acceptable input formats:
            1. time (float) - dt predicted for the same time for all objects
            2. time (list) - dt predicted for each time in list (list dimension must match self.X dimension)
            3. time (tensor) - same as 2
            4. time (list) and idxs (list) of same length - idxs index self.X, all other 
                objects are returned dt if use_default, otherwise returned 0
        
        """
        if self.X is None or len(self.X) == 0:
            return None
        
        if type(target_time) == float: # 1.
            dt = target_time - self.T
            return dt
        
        if type(target_time) == list: # 2 and 4.
            target_time = torch.tensor(target_time,dtype = torch.double) 
            
        if idxs is None:
            dt = target_time - self.T
            return dt
        else:
            #print(len(idxs),len(target_time),self.T.shape[0])

            dt = torch.zeros(len(self.X))
            dt = dt + self.dt_default if use_default else dt
            
            for i in range(len(idxs)):
                internal_idx = self.obj_idxs[idxs[i]]  # we need to switch from obj_id to internal idx
                dt[internal_idx] = target_time[i] - self.T[internal_idx]
            
            return dt
                    
        
    def get_avg_speed(self):
        
        EB_v = self.X[self.D == 1,-1].clone()
        WB_v = self.X[self.D == -1,-1].clone()
        
        # get avg speed
        if len(EB_v) > 0:
            EB_avg = EB_v.mean()
        else:
            EB_avg = self.mu_v / 2
        
        if len(WB_v) > 0:
            WB_avg = WB_v.mean()
        else:
            WB_avg = self.mu_v / 2
            
        return EB_avg,WB_avg
        
    def add(self,detections,obj_ids,directions,times,init_speed = False,classes = None):
        """
        Description
        -----------
        Initializes self.X if this is the first object, otherwise adds new object to X and P 
        
        Parameters
        ----------
        detection - np array of size [n,4] 
            Specifies bounding box x,y,scale and ratio for each detection
        obj_ids - list of length n
            Unique obj_id (int) for each detection
        """
        
        newX = torch.zeros((len(detections),self.state_size)) 
        if len(detections[0]) == self.meas_size:
            try:
                newX[:,:self.meas_size] = torch.from_numpy(detections).to(self.device)
                newD = torch.from_numpy(directions).to(self.device)
                newT = torch.from_numpy(times).to(self.device)
            except:
                newX[:,:self.meas_size] = detections.to(self.device)
                newD = directions.to(self.device)
                newT = times.to(self.device)
                
        else: # case where velocity estimates are given
            try:
                newX = torch.from_numpy(detections).to(device)
                newD = directions.to(self.device)
                newT = times.to(self.device)

            except:
                newX = detections.to(self.device)
                newD = directions.to(self.device)
                newT = times.to(self.device)

        if init_speed:
            
            
            # init speed with mu_v
            newV = self.mu_v.repeat(len(detections)).to(self.device) / 2
            # direction_mask = (newD + 1)  #//2
            # newV = newV/2 + newV/2* direction_mask
            
            # init speed with per-side average
            if True:
                EB_v = []
                WB_v = []
                if self.X is not None and len(self.X) > 0:
                    # get masked speeds per direction 
                    EB_v = self.X[self.D == 1,-1]
                    WB_v = self.X[self.D == -1,-1]
                    
                # get avg speed
                if len(EB_v) > 0:
                    EB_avg = EB_v.mean()
                else:
                    EB_avg = self.mu_v / 2
                
                if len(WB_v) > 0:
                    WB_avg = WB_v.mean()
                else:
                    WB_avg = self.mu_v / 2
                
                # mask speeds per direction and add
                WB = WB_avg.repeat(len(detections)).to(self.device)
                EB = EB_avg.repeat(len(detections)).to(self.device)
                WB_mask = torch.div(directions-1,-2,rounding_mode="trunc")
                EB_mask = torch.div(directions+1,2,rounding_mode="trunc")
                newV = WB*WB_mask + EB*EB_mask
                newX[:,-1] = newV
                
        newP = self.P0.repeat(len(obj_ids),1,1)

        if classes is not None:
            # overwrite l,w,h with class mean values
            for i in range(len(newX)):
                newX[i,2:5] = self.class_size[classes[i]]
                #newP[i,2:5,2:5] = self.class_covariance[classes[i]]
            
            # overwrite l,w,h portion of p with known covariance
        
        # store state and initialize P with defaults
        try:
            new_idx = len(self.X)
            self.X = torch.cat((self.X,newX), dim = 0)
            self.P = torch.cat((self.P,newP), dim = 0)
            self.D = torch.cat((self.D,newD), dim = 0)
            self.T = torch.cat((self.T,newT), dim = 0)
        except:
            new_idx = 0
            self.X = newX.to(self.device).float()
            self.P = newP.to(self.device)
            self.D = newD.to(self.device)
            self.T = newT.to(self.device).double()
            
            
        # add obj_ids to dictionary
        for idx,id in enumerate(obj_ids):
            self.obj_idxs[id] = new_idx
            new_idx = new_idx + 1
        
    
    def remove(self,obj_ids):
        """
        Description
        -----------
        Removes objects indexed by integer id so that they are no longer tracked
        
        Parameters
        ----------
        obj_ids : list of (int) object ids
        """
        if self.X is not None:
            keepers = list(range(len(self.X)))
            for id in obj_ids:
                keepers.remove(self.obj_idxs[id])
                self.obj_idxs[id] = None    
            keepers.sort()
            
            self.X = self.X[keepers,:]
            self.P = self.P[keepers,:]
            self.D = self.D[keepers]
            self.T = self.T[keepers]
            
            # since rows were deleted from X and P, shift idxs accordingly
            new_id = 0
            removals = []
            for id in self.obj_idxs:
                if self.obj_idxs[id] is not None:
                    self.obj_idxs[id] = new_id
                    new_id += 1
                else:
                    removals.append(id)
            for id in removals:
                del self.obj_idxs[id]
    
    def view(self,dt = None,with_direction = False):
        """
        Predicts the state for the given or default dt, but does not update the object states within the filter
        (i.e. non in-place version of predict())
        """
        if self.X is None or len(self.X) == 0:
            return [],torch.empty(0,6)
        
        if dt is None:
            X_pred = self.X
         
        else:
            F_rep = self.F.unsqueeze(0).repeat(len(self.X),1,1)
            F_rep[:,0,5] = self.D * dt
            X_pred = torch.bmm(F_rep,self.X.unsqueeze(2)).squeeze(2)
        
        states = X_pred
        
        inverted = dict([(self.obj_idxs[key],key) for key in self.obj_idxs.keys()])
        id_list = [inverted[i] for i in range(states.shape[0])]
        # get list of IDs - i.e. what obj id does each row correspond to 
        
        if with_direction:
            states = torch.cat((states[:,:-1],self.D.float().unsqueeze(1),states[:,-1:]),dim = 1)
            
        return id_list,states
         
    
    def predict(self,dt = None, classes = None):
        """
        Description:
        -----------
        Uses prediction equations to update X and P without a measurement
        """
        if self.X is None or len(self.X) == 0:
            return
        
        if dt is None:
            dt = self.dt_default
            
        # here we use t and direction. We alter F such that x_dot is signed by direction
        # and corresponds to the timestep t
            
        # update X --> X = XF + mu_F--> [n,7] x [7,7] + [n,7] = [n,7]
        #self.X = torch.mm(self.X,self.F.transpose(0,1)) + self.mu_Q
        F_rep = self.F.unsqueeze(0).repeat(len(self.X),1,1)
        F_rep[:,0,5] = self.D * dt
        
        
        # no model 
        if False:
            F_rep[:,0,5] = 0
            
            
        # scale y uncertainty as a function of speed
        
            
        
        self.X = torch.bmm(F_rep,self.X.unsqueeze(2)).squeeze(2) + self.mu_Q.repeat(len(self.X),1)
        
        
        # update P --> P = FPF^(-1) + Q --> [nx7x7] = [nx7x7] bx [nx7x7] bx [nx7x7] + [n+7x7]
        #F_rep = self.F.unsqueeze(0).repeat(len(self.P),1,1)
        step1 = torch.bmm(F_rep,self.P.float())
        step2 = F_rep.transpose(1,2)
        step3 = torch.bmm(step1,step2)
        
        if classes is not None:
            # classes is a dictionary
            
            cls_vec = torch.zeros(len(self.P))
            for key in classes:
                cls_vec[self.obj_idxs[key]] = classes[key]
            cls_vec = cls_vec.long()
            step4 = self.Q[cls_vec,:,:]
        
        else:
            step4 = self.Q.repeat(len(self.P),1,1)
        
        
        if True:
            step4[:,1,1] *= torch.clamp( self.X[:,5] / 50,min = 0.05, max = 1)
            
        # scale Q by the timestamp, assuming model error is linearly correlated to dt
        # in fact, this should be scaled at with the square root of dt, see https://stats.stackexchange.com/questions/49300/how-does-one-apply-kalman-smoothing-with-irregular-time-steps/585962#585962
        
        # if torch.min(dt) < 0:
        #     print("We got a filtering problem \n\n\n Min dt is less than 0!: {}".format(dt))
        
        
        #dt = torch.clamp(dt,min = 0.001)
        # try:
        #     step4 = step4 * dt/self.dt_default
        # except:
        #     step4 = step4 * dt.unsqueeze(1).unsqueeze(2).repeat(1,self.Q.shape[1],self.Q.shape[2]) / self.dt_default   
        try:
            step4 = step4 * torch.sqrt(dt/self.dt_default)
        except:
            step4 = step4 * torch.sqrt(dt.unsqueeze(1).unsqueeze(2).repeat(1,self.Q.shape[1],self.Q.shape[2]) / self.dt_default)    
            
        self.P = step3 + step4
        self.P = self.P.float()
        
        self.T += dt  # either dt is a single value, or dt is a vector of the same length as self.T
        
        # each item in F_rep[:,0,5] is associated with an obj_id -> we need to get the idea for each
        
        
    def update(self,detections,obj_ids,measurement_idx = 1):
        """
        Description
        -----------
        Updates state for objects corresponding to each obj_id in obj_ids
        Equations taken from: wikipedia.org/wiki/Kalman_filter#Predict
        
        Parameters
        ----------
        detection - np array of size [m,4] 
            Specifies bounding box x,y,scale and ratio for each of m detections
        obj_ids - list of length m
            Unique obj_id (int) for each detection
        """
        
        if measurement_idx == 0:
            mu_R = self.mu_R
            H = self.H
            R = self.R
        elif measurement_idx == 1:
            print("Problem!")
            mu_R = self.mu_R2
            R = self.R2
            H = self.H2
        elif measurement_idx == 2:
            print("Problem!")
            mu_R = self.mu_R3
            R = self.R3
            H = self.H3        
        else:
            print("This measurement index does not exist in this filter")
            raise ValueError
        
        # get relevant portions of X and P
        relevant = [self.obj_idxs[id] for id in obj_ids]
        X_up = self.X[relevant,:]
        P_up = self.P[relevant,:,:].float()
        
        # state innovation --> y = z - XHt --> mx4 = mx4 - [mx7] x [4x7]t  
        try:
            z = torch.from_numpy(detections).to(self.device).double()
        except:
             z = detections.to(self.device).double()
        y = z + mu_R - torch.mm(X_up, H.transpose(0,1))  ######### Not sure if this is right but..
        
        # clamp innovation
        if False:
            clamp = self.y_clamp .unsqueeze(0).expand(y.shape[0],y.shape[1],2)
            y = torch.clamp(y,clamp[:,:,0],clamp[:,:,1])
        
        
        # covariance innovation --> HPHt + R --> [mx4x4] = [mx4x7] bx [mx7x7] bx [mx4x7]t + [mx4x4]
        # where bx is batch matrix multiplication broadcast along dim 0
        # in this case, S = [m,4,4]
        H_rep = H.unsqueeze(0).repeat(len(P_up),1,1)
        step1 = torch.bmm(H_rep,P_up) # this needs to be batched along dim 0
        step2 = torch.bmm(step1,H_rep.transpose(1,2))
        S = step2 + R.repeat(len(P_up),1,1)
        
        # kalman gain --> K = P Ht S^(-1) --> [m,7,4] = [m,7,7] bx [m,7,4]t bx [m,4,4]^-1
        step1 = torch.bmm(P_up,H_rep.transpose(1,2))
        K = torch.bmm(step1,S.inverse())
        
        # A posteriori state estimate --> X_updated = X + Ky --> [mx7] = [mx7] + [mx7x4] bx [mx4x1]
        # must first unsqueeze y to third dimension, then unsqueeze at end
        y = y.unsqueeze(-1).float() # [mx4] --> [mx4x1]
        step1 = torch.bmm(K,y).squeeze(-1) # mx7
        X_up = X_up + step1
        
        # P_updated --> (I-KH)P --> [m,7,7] = ([m,7,7 - [m,7,4] bx [m,4,7]) bx [m,7,7]    
        I = torch.eye(self.state_size).unsqueeze(0).repeat(len(P_up),1,1).to(self.device)
        step1 = I - torch.bmm(K,H_rep)
        P_up = torch.bmm(step1,P_up)
        
        # store updated values
        self.X[relevant,:] = X_up
        self.P[relevant,:,:] = P_up
    
    # def objs(self,with_direction = False):
    #     """
    #     Returns
    #     -------
    #     out_dict - dictionary
    #         Current state of each object indexed by obj_id (int)
    #     """
        
    #     out_dict = {}
    #     for id in self.obj_idxs:
    #         idx = self.obj_idxs[id]
    #         if idx is not None:
    #             out_dict[id] = self.X[idx,:].data.cpu().numpy()
    #     return out_dict        

    def objs(self,with_direction = False,with_time = False):
        """
        Returns
        -------
        out_dict - dictionary
            Current state of each object indexed by obj_id (int)
        """
        
        return self.view(dt = None, with_direction = with_direction)

if __name__ == "__main__":
    """
    A test script in which bounding boxes are randomly generated and jittered to create motion
    """
    
     # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()   
    
    all_trials = [3,10,30,100,300,1000]
    all_results = {"cuda:0":[],"cpu":[]}
    for device in ["cuda:0","cpu"]:
        for n_objs in all_trials:
        
            #n_objs =1000
            n_frames = 1000
            
            ids = list(range(n_objs))
            detections = np.random.rand(n_objs,4)*50
        
            colors = np.random.rand(n_objs,4)
            colors2 = colors.copy()
            colors[:,3]  = 0.2
            colors2[:,3] = 1
            filter = Torch_KF(device)
            
            start_time = time.time()
            
            filter.add(detections,ids)
            for i in range(0,n_frames):
                
                start = time.time()
                filter.predict()
                
                detections = detections + np.random.normal(0,1,[n_objs,4]) + 1
                detections[:,2:] = detections[:,2:]/50
                remove = np.random.randint(0,n_objs - 1)
                
                ids_r = ids.copy()
                del ids_r[remove]
                det_r = detections[ids_r,:]
                start = time.time()
                filter.update(det_r,ids_r)
                tracked_objects = filter.objs()
        
                # if False:
                #     # plot the points to visually confirm that it seems to be working 
                #     x_coords = []
                #     y_coords = []
                #     for key in tracked_objects:
                #         x_coords.append(tracked_objects[key][0])
                #         y_coords.append(tracked_objects[key][1])
                #     for i in range(len(x_coords)):
                #         if i < len(x_coords) -1:
                #             plt.scatter(det_r[i,0],det_r[i,1], color = colors2[i])
                #         plt.scatter(x_coords[i],y_coords[i],s = 300,color = colors[i])
                #         plt.annotate(i,(x_coords[i],y_coords[i]))
                #     plt.draw()
                #     plt.pause(0.0001)
                #     plt.clf()
                
            total_time = time.time() - start_time
            frame_rate = n_frames/total_time
            all_results[device].append(frame_rate)
            print("Filtering {} objects for {} frames took {} sec. Average frame rate: {} on {}".format(n_objs,n_frames,total_time, n_frames/total_time,device))
            torch.cuda.empty_cache()
            
    # plt.figure()   
    # plt.plot(all_trials,all_results['cpu'])
    # plt.plot(all_trials,all_results['cuda:0'])
    # plt.xlabel("Number of filtered objects")
    # plt.ylabel("Frame Rate (Hz)")
    # plt.legend(["CPU","GPU"])
    # plt.title("Frame Rate versus number of objects")