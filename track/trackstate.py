from kf import Torch_KF 
from i24_configparse import parse_cfg
import _pickle as pickle
import torch
import numpy as np
import copy



class TrackState():
    """
    TrackState object maintains the state of the tracking problem. This means storing:
        a. the current state (in a filtering state formulation)
        b. historical states of all active objects
    TrackState objects deal with objects natively in a space-based representation, so 
    object positions are expected in and are returned in state-space coordinates
    
    The following functions are implemented:
        - __call__() wraps dict() or tensor() methods
        - predict() - wrapper around KF predict function
        - update() - wrapper areound KF update that also updates the stored history
        - remove() - removes a set of objects and returns that set
        - add() - adds a set of objects
    """
    
    
    def __init__(self):
        self.fsld = {}                   # fsld[id] stores frames since last detected for object id
        self.all_classes = {}            # dict of lists of integer classes for each object
        self.all_confs = {}              # dict of lists of float confidences for each object
        self._history = {}                # dict of lists, each list item is (time,kf_state) 
        self._next_obj_id = 0             # next id for a new object (incremented during tracking)

        self.class_dict = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi":4,
                    "truck (other)":5,
                    "truck": 5,
                    "motorcycle":6,
                    "trailer":7,
                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi",
                    5:"truck (other)",
                    6:"motorcycle",
                    7:"trailer"
                    }
        
        # load config params (self.device_id, self.kf_param_path,self.n_classes)
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)
        
        self.device = torch.device("cpu") if self.device_id == -1 else torch.device("cuda:{}".format(self.device_id))
        
        if self.kf_param_path is not None:
            with open(self.kf_param_path ,"rb") as f:
                kf_params = pickle.load(f)
        
        if True:
            kf_params["R"] = kf_params["R"].diag().diag()
            kf_params["R"][0,0] = 10/10
            kf_params["R"][1,1] = 2.5/10
            kf_params["R"][2,2] *= 2.25**2
            kf_params["R"][3,3] *= 2.25**2
            kf_params["R"][4,4] *= 1.5**2
            kf_params["mu_R"] *= 0
            
            kf_params["P"] = kf_params["P"].diag().diag()
            kf_params["P"][0,0] = 25
            kf_params["P"][1,1] = 3.5
            kf_params["P"][5,5] = 10e4
    
            # with linear scaling
            # kf_params["Q"] = kf_params["Q"].diag().diag()
            # kf_params["Q"][0,0] /= 4**2
            # kf_params["Q"][3,3] /= 2**2
            # kf_params["Q"][4,4] /= 2**2
            # kf_params["Q"][5,5] /= 15**2
    
            #with sqrt scaling
            kf_params["Q"] = kf_params["Q"][0].diag().diag()
            kf_params["Q"][0,0] = 0.001 #0.003
            kf_params["Q"][1,1] = 0.2 #0.50 #0.75 # just going slightly smaller #1.28 #   based on 9ft/s = 3 standard deviations     0.23 , 20
            kf_params["Q"][2,2] = 0.01
            kf_params["Q"][3,3] = 0.01
            kf_params["Q"][4,4] = 0.01
            kf_params["Q"][5,5] = 16
            
            if self.per_class_model:
                # expand Q to a third dimension with one entry per class
                Q = []
                for i in range(8):
                    temp = copy.deepcopy(kf_params["Q"] )
                    
                    if i == 4:
                        temp[1,1] = 0.05 #0.1
                        #temp[5,5] = 12
                        
                    # elif i == 5:
                    #     temp[1,1] = 0.2
                    #     temp[5,5] = 7
                    Q.append(temp)
                    
                kf_params["Q"] = torch.stack(Q)
    
        #with open("./data/kf_params/kf_params_save4.cpkl","wb") as f:
        #    pickle.dump(kf_params,f)
        
        
        # initialize Kalman filter
        self.kf = Torch_KF(self.device,INIT = kf_params)
        
    def __len__(self):      
        try:
            return self.kf.X.shape[0]
        except AttributeError: # no objects in self.kf.X
            return 0
     
    def __call__(self,target_time = None,with_direction = True,mode = "tensor"):
        """
        returns current state of tracked objects
        :param target_time (float) - time at which the position of objects should be returned
                                  if None, each state is returned at latest time
        :param with_direction (bool) - if True, direction included in state
        :param mode (string) - modulates return type, must be either tensor or dict
        :returns - dict with state keyed by ID, or tensor of IDs and tensor of states
        """
        
        if target_time is not None:
            ids, states = self.kf.view(dt = self.kf.get_dt(target_time),with_direction = with_direction)
        else:
            ids, states = self.kf.view(dt = None,with_direction = with_direction)

        if mode == "tensor":
            return torch.tensor(ids).view(-1), states.view(-1,states.shape[-1])
    
        else:
            state_dict = dict([(ids[i],states[i]) for i in range(len(ids))])
            return state_dict
        
    def add(self,detections,directions,detection_times,labels,scores,init_speed = False):
        """
        detections - [d,5] tensor with x,y,l,w,h
        """
        
        new_ids = []
        
        classes = []
        for i in range(len(detections)):                
            new_ids.append(self._next_obj_id)

            # add internal attributes for new object            
            self.fsld[self._next_obj_id] = 0
            self.all_classes[self._next_obj_id] = np.zeros(self.n_classes)
            self.all_confs[self._next_obj_id] = []
            self._history[self._next_obj_id] = []
            
            # get class and conf for object
            cls = int(labels[i])
            
            if self.use_mean_class_size:
                classes.append(self.class_dict[cls])
                
            self.all_classes[self._next_obj_id][cls] += 1
            self.all_confs[self._next_obj_id].append(scores[i])         
            
            self._next_obj_id += 1
            
            
        if len(detections) > 0:   
            if len(classes) == 0:
                classes = None
            self.kf.add(detections,new_ids,directions,detection_times,init_speed = init_speed,classes = classes)
           
        # update history
        for id in new_ids:
            self._update_history(id)
        
      
    def _update_history(self,id):
        time = self.kf.T[self.kf.obj_idxs[id]].item()
        self._history[id].append((time,self.kf.X[self.kf.obj_idxs[id]]))
        
    def remove(self,ids):
        removals = {}
        for id in ids:
            # store history
            datum = []
            datum.append(self._history.pop(id))
            datum.append(self.all_classes.pop(id))
            removals[id] = datum
        
            del self.all_confs[id]
            del self.fsld[id]
            
        self.kf.remove(ids)
        return removals
    
    def get_classes(self):
        classes = {}
        for id in self.all_classes.keys():
            classes[id] = np.argmax(self.all_classes[id]) 
        return classes
    
    def get_lifespans(self):
        lifespans = {}
        for id in self._history.keys():
            lifespans[id] = len(self._history[id])
        return lifespans
    
    def get_dt(self,target_times,idxs = None):
        return self.kf.get_dt(target_times,idxs = idxs)
    
    def predict(self,dt = None):
        if self.per_class_model:
            classes = self.get_classes()
            self.kf.predict(dt = dt,classes = classes)
        else:
            self.kf.predict(dt = dt)
            
            
    def update(self,detections,obj_ids,classes,confs,measurement_idx = 1,high_confidence_threshold = 0):
        if len(obj_ids) > 0:
            # update kf states
            self.kf.update(detections,obj_ids,measurement_idx = measurement_idx)
        
        # increment all fslds - any obj with an update will have fsld overwritten next
        for id in self.fsld.keys():
            self.fsld[id] += 1 
        
        # update fsld, class, and conf
        for i,id in enumerate(obj_ids):
            self.all_classes[id][int(classes[i])] += 1
            self.all_confs[id].append(confs[i])
            
            if confs[i] > high_confidence_threshold:
                self.fsld[id] = 0
                
        # update stored history
        for id in obj_ids:
            self._update_history(id)
            
        
class TrackierState(TrackState):
    """
    TrackierState object maintains the state of the tracking problem. This means storing:
        a. the current state (in a filtering state formulation)
        b. historical states of all active objects
    TrackState objects deal with objects natively in a space-based representation, so 
    object positions are expected in and are returned in state-space coordinates
    
    The following functions are implemented:
        - __call__() wraps dict() or tensor() methods
        - predict() - wrapper around KF predict function
        - update() - wrapper areound KF update that also updates the stored history
        - remove() - removes a set of objects and returns that set
        - add() - adds a set of objects
        
    BUT WAIT, there's more! It also records the covariance diagonal for each object at each timestamp,
    as well as the output detection confidence
    """
    
    
    def __init__(self):
        super().__init__()
           
      
    def _update_history(self,id,detection = None,prior_cov = None,prior = None):
        
        if detection is None:
            detection = torch.zeros(5)
        if prior_cov is None:
            prior_cov = torch.zeros(6)
        if prior is None:
            prior = torch.zeros(6)
                
        time = self.kf.T[self.kf.obj_idxs[id]].item()
        self._history[id].append((time,self.kf.X[self.kf.obj_idxs[id]].clone(),self.kf.P[self.kf.obj_idxs[id]].diag().clone(),detection,prior_cov,prior))
        
    def remove(self,ids):
        removals = {}
        for id in ids:
            # store history
            datum = []
            datum.append(self._history.pop(id))
            datum.append(self.all_classes.pop(id))
            datum.append(self.all_confs.pop(id))
            removals[id] = datum
        
            del self.fsld[id]
            
        self.kf.remove(ids)
        return removals
    
    
    def update(self,detections,obj_ids,classes,confs,measurement_idx = 0,high_confidence_threshold = 0):
        if len(obj_ids) > 0:
            # update kf states
            
            cache_prior_covs = [self.kf.P[self.kf.obj_idxs[id]].diag().clone() for id in obj_ids]
            cache_priors     = [self.kf.X[self.kf.obj_idxs[id]].clone()        for id in obj_ids]
            
            self.kf.update(detections,obj_ids,measurement_idx = measurement_idx)
        
        # increment all fslds - any obj with an update will have fsld overwritten next
        for id in self.fsld.keys():
            self.fsld[id] += 1 
        
        # update fsld, class, and conf
        for i,id in enumerate(obj_ids):
            self.all_classes[id][int(classes[i])] += 1
            self.all_confs[id].append(confs[i])
            
            if confs[i] > high_confidence_threshold:
                self.fsld[id] = 0
                
        # update stored history
        for idx,id in enumerate(obj_ids):
            self._update_history(id,detections[idx],cache_prior_covs[idx],cache_priors[idx])
        
    
        
