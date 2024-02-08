import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from bbox import im_nms,space_nms, state_nms
from i24_configparse import parse_cfg
from torchvision.ops.boxes import box_iou

def get_Tracker(name):
    """
    getter function that takes a string (class name) input and returns an instance
    of the named class
    """
    if name == "BaseTracker":
        tracker = BaseTracker()
    elif name == "SmartTracker":
        tracker = SmartTracker()  
    else:
        raise NotImplementedError("No BaseTracker child class named {}".format(name))
    
    return tracker

def get_Associator(name,device_id = -1):
    """
    getter function that takes a string (class name) input and returns an instance
    of the named class
    """
    if name == "Associator":
        assoc = Associator()
    elif name == "HungarianIOUAssociator":
        assoc = HungarianIOUAssociator()
    else:
        raise NotImplementedError("No Associator child class named {}".format(name))
    
    return assoc



class Associator():

    def __init__(self):
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)
        
        #self.device = torch.cuda.device("cuda:{}".format(self.device_id) if self.device_id != -1 else "cpu")
        
    def __call__(self,obj_ids,priors,detections,hg = None):
        """
        Applies an object association strategy to pair detections to priors
        :param obj_ids- tensor of length n_objs with unique integer ID for each
        :param priors - tensor of size [n_objs,state_size] with object postions for each
        :param detections - tensor of size [n_detections,state_size] with detection positions for each
        :param hg - HomographyWrapper object
        :return associations - tensor of size n_detections with ID for each detection or -1 if no ID is associated
        """
        
        raise NotImplementedError

class EuclideanAssociator(Associator):
    def __init__(self):
        self = parse_cfg("DEFAULT",obj = self)
        
        #self.device = torch.cuda.device("cuda:{}".format(self.device_id) if self.device_id != -1 else "cpu")
        #print("Matching min IOU: {}".format(self.min_match_iou))
    
    def __call__(self,obj_ids,priors,detections,hg):
       """
       Applies association logic by intersection-over-union metric and Hungarian 
       algorithm for bipartite matching
       
       :param obj_ids- tensor of length n_objs with unique integer ID for each
       :param priors - tensor of size [n_objs,state_size] with object postions for each
       :param detections - tensor of size [n_detections,state_size] with detection positions for each
       :param hg - HomographyWrapper object
       :return associations - tensor of size [n_detections] with ID for each detection or -1 if no ID is associated
       """
       
       # aliases
       first = priors
       second = detections
       
       if len(second) == 0:
            return torch.empty(0)
       elif len(second) == 1:
           second = second.view(1,6)
           
       if len(first) == 0:   
           return torch.zeros(len(second))-1
       elif len(first) == 1:
           first = first.view(1,6)
       
       # keep x and y pos
       first = priors[:,:2].clone()
       second = detections[:,:2].clone()
       
       
       f = first.shape[0]
       s = second.shape[0]
       
       #get weight matrix
       second = second.unsqueeze(0).repeat(f,1,1).double()
       first = first.unsqueeze(1).repeat(1,s,1).double()

       dist = (((first - second)**2).sum(dim = 2)).sqrt()

       try:
           a, b = linear_sum_assignment(dist.data.numpy(),maximize = False) 
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
           if matchings[i] != -1 and  dist[matchings[i],i] > (self.max_dist):
               matchings[i] = -1    
    
        # matchings currently contains object indexes - swap to obj_ids
       try:
            for i in range(len(matchings)):
               if matchings[i] != -1:
                   matchings[i] = obj_ids[matchings[i]]
       except:
           print(type(obj_ids),type(matchings))
           print("Error assigning obj_ids to matchings. len matchings: {}, len obj_ids: {}".format(matchings.shape,obj_ids.shape))
           return torch.zeros(len(second))-1
                   
       return torch.from_numpy(matchings)
        

        
        
class HungarianIOUAssociator(Associator):
    def __init__(self):
        self = parse_cfg("DEFAULT",obj = self)
        
        #self.device = torch.cuda.device("cuda:{}".format(self.device_id) if self.device_id != -1 else "cpu")
        #print("Matching min IOU: {}".format(self.min_match_iou))
    
    def __call__(self,obj_ids,priors,detections,hg):
       """
       Applies association logic by intersection-over-union metric and Hungarian 
       algorithm for bipartite matching
       
       :param obj_ids- tensor of length n_objs with unique integer ID for each
       :param priors - tensor of size [n_objs,state_size] with object postions for each
       :param detections - tensor of size [n_detections,state_size] with detection positions for each
       :param hg - HomographyWrapper object
       :return associations - tensor of size [n_detections] with ID for each detection or -1 if no ID is associated
       """
       
       # aliases
       first = priors
       second = detections
       
       if len(second) == 0:
            return torch.empty(0)
       elif len(second) == 1:
           second = second.view(1,6)
           
       if len(first) == 0:   
           return torch.zeros(len(second))-1
       elif len(first) == 1:
           first = first.view(1,6)

        
       # print("Priors:")
       # print(first)
       # print("Detections:")
       # print(second)

        
        

       # first and second are in state form - convert to space form
       # first = hg.state_to_space(first.clone())
       # boxes_new = torch.zeros([first.shape[0],4],device = first.device)
       # boxes_new[:,0] = torch.min(first[:,0:4,0],dim = 1)[0]
       # boxes_new[:,2] = torch.max(first[:,0:4,0],dim = 1)[0]
       # boxes_new[:,1] = torch.min(first[:,0:4,1],dim = 1)[0]
       # boxes_new[:,3] = torch.max(first[:,0:4,1],dim = 1)[0]
       # first = boxes_new
       
       # second = hg.state_to_space(second.clone())
       # boxes_new = torch.zeros([second.shape[0],4],device = second.device)
       # boxes_new[:,0] = torch.min(second[:,0:4,0],dim = 1)[0]
       # boxes_new[:,2] = torch.max(second[:,0:4,0],dim = 1)[0]
       # boxes_new[:,1] = torch.min(second[:,0:4,1],dim = 1)[0]
       # boxes_new[:,3] = torch.max(second[:,0:4,1],dim = 1)[0]
       # second = boxes_new
       
       # convert from state form to state-space bbox form
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
       dist = self.md_iou(first,second)

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
           if matchings[i] != -1 and  dist[matchings[i],i] < (self.min_match_iou):
               matchings[i] = -1    
    
        # matchings currently contains object indexes - swap to obj_ids
       try:
            for i in range(len(matchings)):
               if matchings[i] != -1:
                   matchings[i] = obj_ids[matchings[i]]
       except:
           print(type(obj_ids),type(matchings))
           print("Error assigning obj_ids to matchings. len matchings: {}, len obj_ids: {}".format(matchings.shape,obj_ids.shape))
           return torch.zeros(len(second))-1
                   
       # print("Dist:")
       # print(dist)
       # print("Matches:")
       # print(matchings)
        
       return torch.from_numpy(matchings)
    
            
   
    def md_iou(self,a,b):
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
        
        #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
        return iou



class ByteIOUAssociator(HungarianIOUAssociator):

    def __call__(self,obj_ids,priors,detections,hg,det_cam_names = None,obj_cam_names = None):
        """
        Applies association logic by intersection-over-union metric and Hungarian 
        algorithm for bipartite matching
        
        :param obj_ids- tensor of length n_objs with unique integer ID for each
        :param priors - tensor of size [n_objs,state_size] with object postions for each
        :param detections - tensor of size [n_detections,state_size+1] with detection positions for each, with confidence appended as -1 to each detection
        :param hg - HomographyWrapper object
        :param camera_idxs - None or tensor of length n
        :return associations - tensor of size [n_detections] with ID for each detection or -1 if no ID is associated
        
        But, unlike base IOU tracking, matching is performed in 2 steps
        
        i.) Match objects to high confidence detections
        ii.) Match remaining objects to low-confidence detections (i.e. something is better than nothing)
        iii.) Take unmatched high-confidence detections as new objects
        iv.) They suggest IOU  = 0.2, high conf threshold = 0.6
        """

        # aliases
        first = priors.clone()
        second = detections.clone()

        confs = detections[:,-1]
        
        if len(second) == 0:
             return torch.empty(0)
        
        if len(first) == 0:   
            return torch.zeros(len(second))-1
        
        # print("Priors:")
        # print(first)
        # print("Detections:")
        # print(second)

        #  subdivide detections into low and high confidence pools
        high_idx = torch.where(confs >= self.tau_high, 1,0).nonzero().squeeze()
        low_idx = torch.where(confs < self.tau_high, 1,0).nonzero().squeeze()
        
        third = second[low_idx,:]
        second = second[high_idx,:]
        
        if third.ndim == 1:
            third = third.unsqueeze(0)
        if second.ndim ==1:
            second = second.unsqueeze(0)
        
        # likewise, have to subdivide det_cam_names 
        if det_cam_names is not None:
            det_cam_names_high = [det_cam_names[idx] for idx in high_idx]
            det_cam_names_low  = [det_cam_names[idx] for idx in low_idx]
        else:
            det_cam_names_high = None
            det_cam_names_low  = None
        
        # use high confidence pool for first stage matching

        boxes = first
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
        boxes = second
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
        dist = self.md_iou(first,second)
        
        if det_cam_names_high is not None and obj_cam_names is not None:
            
            for i in range(len(det_cam_names_high)):
                for j in range(len(obj_cam_names)):
                    if obj_cam_names[j] != det_cam_names_high[i]:
                        dist[j,i] = 0
        
       
                    

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
            if matchings[i] != -1 and  dist[matchings[i],i] < (self.min_match_iou):
                matchings[i] = -1    
     
         # matchings currently contains object indexes - swap to obj_ids
        for i in range(len(matchings)):
            if matchings[i] != -1:
                matchings[i] = obj_ids[matchings[i]]
                






         
        # at this point matchings is a tensor of shape[d], where matchings[didx] is -1 if that detection wasn't matched to anyone, id otherwise
        
        # next, we get the set of object ids and priors that aren't matched yet
        unmatched_ids = []
        unmatched_idxs = []
        for idx,id in enumerate(obj_ids):
            id = id.item()
            if id not in matchings:
                unmatched_ids.append(id)
                unmatched_idxs.append(idx)
        
        if obj_cam_names is not None:
            unmatched_obj_cam_names = [obj_cam_names[idx] for idx in unmatched_idxs]
        first = priors.clone()
        unmatched_priors = first[unmatched_idxs,:]        
        
        boxes = unmatched_priors.clone()
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
        boxes = third
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
        third = boxes_new.clone()
        
        
        f = first.shape[0]
        s = third.shape[0]
        
        #get weight matrix
        third = third.unsqueeze(0).repeat(f,1,1).double()
        first = first.unsqueeze(1).repeat(1,s,1).double()
        dist = self.md_iou(first,third)
        
        if det_cam_names_high is not None and obj_cam_names is not None:
            
            for i in range(len(det_cam_names_low)):
                for j in range(len(unmatched_obj_cam_names)):
                    if unmatched_obj_cam_names[j] != det_cam_names_low[i]:
                        dist[j,i] = 0
        
        try:
            a, b = linear_sum_assignment(dist.data.numpy(),maximize = True) 
        except ValueError:
             return torch.zeros(s)-1
             print("DEREK USE LOGGER WARNING HERE")
         
        
        # convert into expected form
        second_matchings = np.zeros(s)-2
        for idx in range(0,len(b)):
             second_matchings[b[idx]] = a[idx]
        second_matchings = np.ndarray.astype(second_matchings,int)
         
        # remove any matches too far away
        for i in range(len(second_matchings)):
            if second_matchings[i] != -2 and  dist[second_matchings[i],i] < (self.min_match_iou):
                second_matchings[i] = -2    
     
         # matchings currently contains object indexes - swap to obj_ids
        for i in range(len(second_matchings)):
            if second_matchings[i] != -2:
                second_matchings[i] = unmatched_ids[second_matchings[i]]


        # now we have matchings of length len(high_idx) and second_matchings of length len(low_idx)
        # matchings has -1 if no match, and second_matchings has -2 if no match
        # we need to re-interleave them
        final_matchings = torch.zeros(detections.shape[0],dtype = int)
        final_matchings[high_idx] = torch.from_numpy(matchings)
        final_matchings[low_idx] = torch.from_numpy(second_matchings)
        final_matchings = torch.clamp(final_matchings,min = -1)
        
        return final_matchings


class ByteEucAssociator(Associator):

    def __call__(self,obj_ids,priors,detections,hg,det_cam_names = None,obj_cam_names = None):
        """
        Applies association logic by intersection-over-union metric and Hungarian 
        algorithm for bipartite matching
        
        :param obj_ids- tensor of length n_objs with unique integer ID for each
        :param priors - tensor of size [n_objs,state_size] with object postions for each
        :param detections - tensor of size [n_detections,state_size+1] with detection positions for each, with confidence appended as -1 to each detection
        :param hg - HomographyWrapper object
        :param camera_idxs - None or tensor of length n
        :return associations - tensor of size [n_detections] with ID for each detection or -1 if no ID is associated
        
        But, unlike base IOU tracking, matching is performed in 2 steps
        
        i.) Match objects to high confidence detections
        ii.) Match remaining objects to low-confidence detections (i.e. something is better than nothing)
        iii.) Take unmatched high-confidence detections as new objects
        iv.) They suggest IOU  = 0.2, high conf threshold = 0.6
        """

        # aliases
        first = priors[:,:2].clone()
        second = detections[:,:2].clone()

        confs = detections[:,-1]
        
        if len(second) == 0:
             return torch.empty(0)
        
        if len(first) == 0:   
            return torch.zeros(len(second))-1
        
        # print("Priors:")
        # print(first)
        # print("Detections:")
        # print(second)

        #  subdivide detections into low and high confidence pools
        high_idx = torch.where(confs >= self.tau_high, 1,0).nonzero().squeeze()
        low_idx = torch.where(confs < self.tau_high, 1,0).nonzero().squeeze()
        
        third = second[low_idx,:]
        second = second[high_idx,:]
        
        if third.ndim == 1:
            third = third.unsqueeze(0)
        if second.ndim ==1:
            second = second.unsqueeze(0)
        
        # likewise, have to subdivide det_cam_names 
        if det_cam_names is not None:
            det_cam_names_high = [det_cam_names[idx] for idx in high_idx]
            det_cam_names_low  = [det_cam_names[idx] for idx in low_idx]
        else:
            det_cam_names_high = None
            det_cam_names_low  = None
        
        # use high confidence pool for first stage matching

        # boxes = first
        # d = boxes.shape[0]
        # intermediate_boxes = torch.zeros([d,4,2], device = boxes.device)
        # intermediate_boxes[:,0,0] = boxes[:,0] 
        # intermediate_boxes[:,0,1] = boxes[:,1] - boxes[:,3]/2.0
        # intermediate_boxes[:,1,0] = boxes[:,0] 
        # intermediate_boxes[:,1,1] = boxes[:,1] + boxes[:,3]/2.0
        
        # intermediate_boxes[:,2,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
        # intermediate_boxes[:,2,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] - boxes[:,3]/2.0
        # intermediate_boxes[:,3,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
        # intermediate_boxes[:,3,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] + boxes[:,3]/2.0

        # boxes_new = torch.zeros([boxes.shape[0],4],device = boxes.device)
        # boxes_new[:,0] = torch.min(intermediate_boxes[:,0:4,0],dim = 1)[0]
        # boxes_new[:,2] = torch.max(intermediate_boxes[:,0:4,0],dim = 1)[0]
        # boxes_new[:,1] = torch.min(intermediate_boxes[:,0:4,1],dim = 1)[0]
        # boxes_new[:,3] = torch.max(intermediate_boxes[:,0:4,1],dim = 1)[0]
        # first = boxes_new.clone()
        
        # # convert from state form to state-space bbox form
        # boxes = second
        # d = boxes.shape[0]
        # intermediate_boxes = torch.zeros([d,4,2], device = boxes.device)
        # intermediate_boxes[:,0,0] = boxes[:,0] 
        # intermediate_boxes[:,0,1] = boxes[:,1] - boxes[:,3]/2.0
        # intermediate_boxes[:,1,0] = boxes[:,0] 
        # intermediate_boxes[:,1,1] = boxes[:,1] + boxes[:,3]/2.0
        
        # intermediate_boxes[:,2,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
        # intermediate_boxes[:,2,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] - boxes[:,3]/2.0
        # intermediate_boxes[:,3,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
        # intermediate_boxes[:,3,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] + boxes[:,3]/2.0

        # boxes_new = torch.zeros([boxes.shape[0],4],device = boxes.device)
        # boxes_new[:,0] = torch.min(intermediate_boxes[:,0:4,0],dim = 1)[0]
        # boxes_new[:,2] = torch.max(intermediate_boxes[:,0:4,0],dim = 1)[0]
        # boxes_new[:,1] = torch.min(intermediate_boxes[:,0:4,1],dim = 1)[0]
        # boxes_new[:,3] = torch.max(intermediate_boxes[:,0:4,1],dim = 1)[0]
        # second = boxes_new.clone()
        
        f = first.shape[0]
        s = second.shape[0]
        
        #get weight matrix
        second = second.unsqueeze(0).repeat(f,1,1).double()
        first = first.unsqueeze(1).repeat(1,s,1).double()
        dist = (((first - second)**2).sum(dim = 2)).sqrt()
        
        if det_cam_names_high is not None and obj_cam_names is not None:
            
            for i in range(len(det_cam_names_high)):
                for j in range(len(obj_cam_names)):
                    if obj_cam_names[j] != det_cam_names_high[i]:
                        dist[j,i] = 0
        
       
                    

        try:
            a, b = linear_sum_assignment(dist.data.numpy(),maximize = False) 
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
            if matchings[i] != -1 and  dist[matchings[i],i] > (self.max_dist):
                matchings[i] = -1    
     
         # matchings currently contains object indexes - swap to obj_ids
        for i in range(len(matchings)):
            if matchings[i] != -1:
                matchings[i] = obj_ids[matchings[i]]
                






         
        # at this point matchings is a tensor of shape[d], where matchings[didx] is -1 if that detection wasn't matched to anyone, id otherwise
        
        # next, we get the set of object ids and priors that aren't matched yet
        unmatched_ids = []
        unmatched_idxs = []
        for idx,id in enumerate(obj_ids):
            id = id.item()
            if id not in matchings:
                unmatched_ids.append(id)
                unmatched_idxs.append(idx)
        
        if obj_cam_names is not None:
            unmatched_obj_cam_names = [obj_cam_names[idx] for idx in unmatched_idxs]
        first = priors[:,:2].clone()
        unmatched_priors = first[unmatched_idxs,:]        
        
        first = unmatched_priors.clone()
        # d = boxes.shape[0]
        # intermediate_boxes = torch.zeros([d,4,2], device = boxes.device)
        # intermediate_boxes[:,0,0] = boxes[:,0] 
        # intermediate_boxes[:,0,1] = boxes[:,1] - boxes[:,3]/2.0
        # intermediate_boxes[:,1,0] = boxes[:,0] 
        # intermediate_boxes[:,1,1] = boxes[:,1] + boxes[:,3]/2.0
        
        # intermediate_boxes[:,2,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
        # intermediate_boxes[:,2,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] - boxes[:,3]/2.0
        # intermediate_boxes[:,3,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
        # intermediate_boxes[:,3,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] + boxes[:,3]/2.0

        # boxes_new = torch.zeros([boxes.shape[0],4],device = boxes.device)
        # boxes_new[:,0] = torch.min(intermediate_boxes[:,0:4,0],dim = 1)[0]
        # boxes_new[:,2] = torch.max(intermediate_boxes[:,0:4,0],dim = 1)[0]
        # boxes_new[:,1] = torch.min(intermediate_boxes[:,0:4,1],dim = 1)[0]
        # boxes_new[:,3] = torch.max(intermediate_boxes[:,0:4,1],dim = 1)[0]
        # first = boxes_new.clone()
        
        # convert from state form to state-space bbox form
        # boxes = third
        # d = boxes.shape[0]
        # intermediate_boxes = torch.zeros([d,4,2], device = boxes.device)
        # intermediate_boxes[:,0,0] = boxes[:,0] 
        # intermediate_boxes[:,0,1] = boxes[:,1] - boxes[:,3]/2.0
        # intermediate_boxes[:,1,0] = boxes[:,0] 
        # intermediate_boxes[:,1,1] = boxes[:,1] + boxes[:,3]/2.0
        
        # intermediate_boxes[:,2,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
        # intermediate_boxes[:,2,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] - boxes[:,3]/2.0
        # intermediate_boxes[:,3,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
        # intermediate_boxes[:,3,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] + boxes[:,3]/2.0

        # boxes_new = torch.zeros([boxes.shape[0],4],device = boxes.device)
        # boxes_new[:,0] = torch.min(intermediate_boxes[:,0:4,0],dim = 1)[0]
        # boxes_new[:,2] = torch.max(intermediate_boxes[:,0:4,0],dim = 1)[0]
        # boxes_new[:,1] = torch.min(intermediate_boxes[:,0:4,1],dim = 1)[0]
        # boxes_new[:,3] = torch.max(intermediate_boxes[:,0:4,1],dim = 1)[0]
        # third = boxes_new.clone()
        
        
        f = first.shape[0]
        s = third.shape[0]
        
        #get weight matrix
        third = third.unsqueeze(0).repeat(f,1,1).double()
        first = first.unsqueeze(1).repeat(1,s,1).double()
        dist = (((first - third)**2).sum(dim = 2)).sqrt()        
        
        if det_cam_names_high is not None and obj_cam_names is not None:
            
            for i in range(len(det_cam_names_low)):
                for j in range(len(unmatched_obj_cam_names)):
                    if unmatched_obj_cam_names[j] != det_cam_names_low[i]:
                        dist[j,i] = 0
        
        try:
            a, b = linear_sum_assignment(dist.data.numpy(),maximize = False) 
        except ValueError:
             return torch.zeros(s)-1
             print("DEREK USE LOGGER WARNING HERE")
         
        
        # convert into expected form
        second_matchings = np.zeros(s)-2
        for idx in range(0,len(b)):
             second_matchings[b[idx]] = a[idx]
        second_matchings = np.ndarray.astype(second_matchings,int)
         
        # remove any matches too far away
        for i in range(len(second_matchings)):
            if second_matchings[i] != -2 and  dist[second_matchings[i],i] > (self.max_dist):
                second_matchings[i] = -2    
     
         # matchings currently contains object indexes - swap to obj_ids
        for i in range(len(second_matchings)):
            if second_matchings[i] != -2:
                second_matchings[i] = unmatched_ids[second_matchings[i]]


        # now we have matchings of length len(high_idx) and second_matchings of length len(low_idx)
        # matchings has -1 if no match, and second_matchings has -2 if no match
        # we need to re-interleave them
        final_matchings = torch.zeros(detections.shape[0],dtype = int)
        final_matchings[high_idx] = torch.from_numpy(matchings)
        final_matchings[low_idx] = torch.from_numpy(second_matchings)
        final_matchings = torch.clamp(final_matchings,min = -1)
        
        return final_matchings

class BaseTracker():
    """
    Basic wrapper around TrackState that adds some basic functionality for object 
    management and stale object removal. The thinking was that this may be nonstandard
    so should be abstracted away from the TrackState object, but the two classes
    may be unified in a future version.
    """
    
    def __init__(self):
        
        # parse config file
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)    
        

    def preprocess(self,tstate,obj_times):
        """
        Receives a TrackState object as input, as well as the times for each object
        Applies kf.predict to objects
        Theoretically, additional logic could happen here i.e. a subset of objects
        could be selected for update based on covarinace, etc.
        
        :param tstate - TrackState object
        :param obj_times - tensor of times of same length as number of active objects
        
        :return obj_ids - tensor of int IDs for object priors
        :return priors - tensor of all TrackState object priors
        :return selected_idxs - tensor of idxs on which to perform measurement update (naively, all of them)
        """
        
        #dts = tstate.get_dt(obj_times)
        #tstate.predict(dt = dts)   
        
        obj_ids,priors = tstate(target_time = obj_times)
        
        selected_idxs = self.select_idxs(tstate,obj_times)
        
        return obj_ids,priors,selected_idxs
        
    def select_idxs(self,tstate,obj_times):
        return torch.tensor([i for i in range(len(tstate))])
        

    def postprocess(self,detections,detection_times,classes,confs,assigned_ids,tstate,hg = None,measurement_idx = 0):
        """
        Updates KF representation of objects where assigned_id is not -1 (unassigned)
        Adds other objects as new detections
        For all TrackState objects, checks confidences and fslds and removes inactive objects
        :param detections -tensor of size [n_detections,state_size]
        :param detection_times - tensor of size [n_detections] with frame time
        :param classes - tensor of size [n_detections] with integer class prediction for each
        :param confs - tensor of size [n_detections] of confidences in range[0,1]
        :param assigned_ids - tensor of size [n_detections] of IDs, or -1 if no id assigned
        :param tstate - TrackState object
        :param meas_idx - int specifying which measurement type was used
        
        :return - stale_objects - dictionary of object histories indexed by object ID
        """
        
        # get IDs and times for update
        if len(assigned_ids) > 0:
            update_idxs = torch.nonzero(assigned_ids + 1).squeeze(1) 
            
            if len(update_idxs) > 0:
                update_ids = assigned_ids[update_idxs].tolist()
                update_times = detection_times[update_idxs]
                
                tstate_ids = tstate()[0]
                for id in update_ids:
                    assert (id in tstate_ids), "{}".format(id)
                    
                

                # TODO this may give an issue when some but not all objects need to be rolled forward
                # roll existing objects forward to the detection times
                dts = tstate.get_dt(update_times,idxs = update_ids)
                tstate.predict(dt = dts)
            
            
                # update assigned detections
                update_detections = detections[update_idxs,:]
                update_classes = classes[update_idxs]
                update_confs = confs[update_idxs]
                tstate.update(update_detections[:,:5],update_ids,update_classes,update_confs, measurement_idx = measurement_idx)
            
            # collect unassigned detections
            new_idxs = [i for i in range(len(assigned_ids))]
            for i in update_idxs:
                new_idxs.remove(i)
              
            
            # add new detections as new objects
            if len(new_idxs) > 0:
                new_idxs = torch.tensor(new_idxs)
                new_detections = detections[new_idxs,:]
                new_classes = classes[new_idxs]
                new_confs = confs[new_idxs]
                new_times = detection_times[new_idxs]
                
                
                # # do nms across all device batches to remove dups
                # if hg is not None:
                #     space_new = hg.state_to_space(new_detections)
                #     keep = space_nms(space_new,new_confs)
                #     new_detections = detections[keep,:]
                #     new_classes = classes[keep]
                #     new_confs = confs[keep]
                #     new_times = detection_times[keep]
                
                # create direction tensor based on location
                directions = torch.where(new_detections[:,1] > 60, torch.zeros(new_confs.shape)-1,torch.ones(new_confs.shape))
                
                tstate.add(new_detections,directions,new_times,new_classes,new_confs,init_speed = True)
            
            
            # do nms on all objects to remove overlaps, where score = # of frames since initialized
            if  hg is not None:
                ids,states = tstate()
                space = hg.state_to_space(states)
                lifespans = tstate.get_lifespans()
                scores = torch.tensor([lifespans[id.item()] for id in ids])
                keep = space_nms(space,scores.float(),threshold = 0.1)
                keep_ids = ids[keep]
                
                removals = ids.tolist()
                for id in keep_ids:
                    removals.remove(id)
                    
                tstate.remove(removals)
                
            # remove anomalies
            ids,states = tstate()
            removals = []
            for i in range(len(ids)):
                if states[i][2] > 80 or states[i][2] < 10 or states[i][3] > 15 or states[i][3] < 3 or states[i][4] > 16 or states[i][4] < 3:
                    removals.append(ids[i].item())
            tstate.remove(removals)
            
          
        # if no detections, increment fsld in all tracked objects
        else:
            tstate.update(None,[],None,None)
        
        # remove objects
        stale_objects = self.remove(tstate)
        
        return stale_objects


    # TODO - remove this hard-coded removal rule
    def remove(self,tstate):
        """
        
        """
        # remove stale objects
        removals = []
        ids,states = tstate()
        for i in range(len(ids)):
            id = ids[i].item()
            if tstate.fsld[id] > self.fsld_max or states[i][0] < -200 or states[i][0] > 1800:
                removals.append(id)
                
        stale_objects = tstate.remove(removals)
        return stale_objects
    
    def flush(self,tstate):
        """
        Removes all objects from tracker
        """
        ids,states = tstate()
        stale_objects = tstate.remove(ids)
        return stale_objects
    
 
class SmartTracker(BaseTracker):
    
    def postprocess(self,
                    detections,
                    detection_times,
                    classes,
                    confs,
                    assigned_ids,
                    tstate,
                    hg = None,
                    measurement_idx = 0):
        """
        Updates KF representation of objects where assigned_id is not -1 (unassigned)
        Adds other objects as new detections
        For all TrackState objects, checks confidences and fslds and removes inactive objects
        :param detections -tensor of size [n_detections,state_size]
        :param detection_times - tensor of size [n_detections] with frame time
        :param classes - tensor of size [n_detections] with integer class prediction for each
        :param confs - tensor of size [n_detections] of confidences in range[0,1]
        :param assigned_ids - tensor of size [n_detections] of IDs, or -1 if no id assigned
        :param tstate - TrackState object
        :param meas_idx - int specifying which measurement type was used
        
        :return - stale_objects - dictionary of object histories indexed by object ID
        """

        # get IDs and times for update
        if len(assigned_ids) > 0:
            update_idxs = torch.nonzero(assigned_ids + 1).squeeze(1) 
            
            if len(update_idxs) > 0:
                update_ids = assigned_ids[update_idxs].tolist()
                update_times = detection_times[update_idxs]
                
                tstate_ids = tstate()[0]
                for id in update_ids:
                    assert (id in tstate_ids), "{}".format(id)
                    
                

                # TODO this may give an issue when some but not all objects need to be rolled forward
                # roll existing objects forward to the detection times
                dts = tstate.get_dt(update_times,idxs = update_ids)
                tstate.predict(dt = dts)
            
            
                # update assigned detections
                update_detections = detections[update_idxs,:]
                update_classes = classes[update_idxs]
                update_confs = confs[update_idxs]
                
                if False:
                    lifespans = tstate.get_lifespans()
                    update_lifespans = torch.tensor([lifespans[id] for id in update_ids])
                    update_detections = self.de_overlap(update_detections,update_lifespans, hg)
                
                tstate.update(update_detections[:,:5],update_ids,update_classes,update_confs, measurement_idx = measurement_idx,high_confidence_threshold = self.sigma_high)
            
            # collect unassigned detections
            new_idxs = [i for i in range(len(assigned_ids))]
            for i in update_idxs:
                new_idxs.remove(i)
              
            # if initializations is not None:
            #     if len(initializations) > 0:
            #         #stack initializations as tensor
            #         new_detections = torch.stack( [torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"]]) for obj in initializations])
            #         new_classes    = torch.tensor( [hg.hg1.class_dict[obj["class"]]                                  for obj in initializations])
            #         new_confs      = torch.ones ( len(initializations))
            #         new_times      = torch.tensor( [obj["timestamp"]                                             for obj in initializations])
                    
            #         # create direction tensor based on location
            #         directions = torch.where(new_detections[:,1] > 60, torch.zeros(new_confs.shape)-1,torch.ones(new_confs.shape))
            #         tstate.add(new_detections,directions,new_times,new_classes,new_confs,init_speed = True)
                
            # add new detections as new objects
            if len(new_idxs) > 0:
                new_idxs = torch.tensor(new_idxs)
                new_detections = detections[new_idxs,:]
                new_classes = classes[new_idxs]
                new_confs = confs[new_idxs]
                new_times = detection_times[new_idxs]
                
                # create direction tensor based on location
                #directions = torch.where(new_detections[:,1] > 60, torch.zeros(new_confs.shape)-1,torch.ones(new_confs.shape))
                directions = new_detections[:,5]
                tstate.add(new_detections[:,:5],directions,new_times,new_classes,new_confs,init_speed = True)
                
        # if no detections, increment fsld in all tracked objects
        else:
            tstate.update(None,[],None,None)
            
        stale_objects = self.remove(tstate,hg)
        return stale_objects
            
    def medianize(self,tstate,obj_ids):
        """
        Re-initialize object state (expect x-position) with median of all historical states so far,
        then re-initialize state covariance lower
        """
        
        for id in obj_ids:
            
            # get history
            hist = tstate._history[id]
            
            # get median state
            states = []
            for item in hist:
                states.append(item[1])
            states = torch.stack(states)
            median = torch.median(states,dim = 0)[0]
            
            # overwrite state with median
            tstate.kf.X[tstate.kf.obj_idxs[id],1:] = median [1:]
            
            # overwrite covariance with new covariance
            tstate.kf.P[tstate.kf.obj_idxs[id],1:,1:] /= 1
    
    def remove(self,tstate,hg = None):
            
            out_objs = {}
            COD = {}
            
            
            # 5. Remove lost objects (too many frames since last detected)    
            if self.fsld_max != -1:
                removals = []
                ids,states = tstate()  
                for i in range(len(ids)):
                    id = ids[i].item()
                    if tstate.fsld[id] > self.fsld_max:
                        removals.append(id)
                if len(removals) > 0:
                    objs = tstate.remove(removals)
                    for key in objs.keys():
                        out_objs[key] = objs[key] 
                        COD[key] = "Lost"
                    
            # 1. do nms on all objects to remove overlaps, where score = # of frames since initialized
            if hg is not None and len(tstate) > 0:
                ids,states = tstate(target_time = tstate.kf.T[0])  # so objects are at same time??
                #space = hg.state_to_space(states)
                lifespans = tstate.get_lifespans()
                scores = torch.tensor([lifespans[id.item()] for id in ids])
                keep = state_nms(states,scores.float(),threshold =self.iou_max)
                keep_ids = ids[keep]
                
                removals = ids.tolist()
                for id in keep_ids:
                    removals.remove(id)
                
                objs = tstate.remove(removals)
                for key in objs.keys():
                    out_objs[key] = objs[key] 
                    COD[key] = "Overlap"
                    
            
                  
            # 2. Remove anomalies
            ids,states = tstate()
            removals = []
            for i in range(len(ids)):
                for j in range(5):
                    if states[i,j] < self.state_bounds[2*j] or states[i,j] > self.state_bounds[2*j+1]:
                        removals.append(ids[i].item())
                        break
            objs = tstate.remove(removals)
            for key in objs.keys():
                out_objs[key] = objs[key] 
                COD[key] = "Anomalous state"

            
            # 3. Remove objects that don't have enough high confidence detections
            ids,states = tstate()
            removals = []
            
            high_ids = [] # to be medianized
            for id in ids:
                id = id.item()
                if len(tstate.all_confs[id]) == self.n_init:
                    high = True
                    for conf in tstate.all_confs[id]:
                        if conf < self.sigma_high:
                            removals.append(id)
                            high = False
                            break
                    if high:
                        high_ids.append(id)
                        
            objs = tstate.remove(removals)
            for key in objs.keys():
                out_objs[key] = objs[key] 
                COD[key] = "Low confidence"
                
            if self.median_initialize:
                self.medianize(tstate,high_ids)

            # 4. Pop objects that are out of FOV
            removals = []
            ids,states = tstate()
            for i in range(len(ids)):
                id = ids[i].item()
                if  states[i][0] < self.fov[0] or states[i][0] > self.fov[1]:
                    removals.append(id)
                    
            objs = tstate.remove(removals)
            for key in objs.keys():
                out_objs[key] = objs[key] 
                COD[key] = "Exit FOV"
                
           
            
            return out_objs,COD
        
    def flush(self,tstate):
        """
        Removes all objects from tracker
        """
        COD = {}
        ids,states = tstate()
        ids = [id.item() for id in ids]
        stale_objects = tstate.remove(ids)
        for id in ids:
            COD[id] = "Active at End"
        return stale_objects, COD
    
    
    def de_overlap(ids,boxes,lifespans,hg, x_buffer = 2 , y_buffer = 1):
                         
         
            
          first = hg.state_to_space(boxes.clone())
          boxes_new = torch.zeros([first.shape[0],4],device = first.device)
          boxes_new[:,0] = torch.min(first[:,0:4,0],dim = 1)[0] - x_buffer
          boxes_new[:,2] = torch.max(first[:,0:4,0],dim = 1)[0] - y_buffer
          boxes_new[:,1] = torch.min(first[:,0:4,1],dim = 1)[0] + x_buffer
          boxes_new[:,3] = torch.max(first[:,0:4,1],dim = 1)[0] + y_buffer
          
          # calc x overlap and y overlap for all pairs
          bn0 = boxes_new.shape[0]
          a = boxes_new.unsqueeze(0).expand(bn0,bn0,4)
          b = boxes_new.unsqueeze(1).expand(bn0,bn0,4)
          
          # x_overlap  - shift of box a in + direction required to remove overlap
          x_overlap =   b[:,:,2] - a[:,:,0]
          x_overlap = torch.clamp(x_overlap,min = 0)
    
          # x_overlap2 - shift of box a in - direction required to remove overlap (so values are negative)
          x_overlap2 =  b[:,:,0] - a[:,:,2]
          x_overlap2 = torch.clamp(x_overlap2,max = 0)
    
          # y_overlap - shift of box a in + direction required to remove overlap
          y_overlap = b[:,:,3] - a[:,:,1]
          y_overlap = torch.clamp(y_overlap,min = 0)
          
          y_overlap2 = b[:,:,1] - a[:,:,3]
          y_overlap2 = torch.clamp(y_overlap2,max = 0)
    
          ## any of these shifts will fix the overlap, so we simply need to select the minimum magnitude shift to fix the problem       
          
          
          
          x_shifts = torch.where(x_overlap < torch.abs(x_overlap2),x_overlap,x_overlap2)
          y_shifts = torch.where(y_overlap < torch.abs(y_overlap2),y_overlap,y_overlap2)
          
          x_shifts *= (1-torch.eye(bn0))
          y_shifts *= (1-torch.eye(bn0))
          
          x_mask = torch.where(torch.abs(x_shifts) < torch.abs(y_shifts),1,0)
          
          x_shifts *= x_mask
          y_shifts *= (1-x_mask)
          
          a_lifespans = lifespans.unsqueeze(0).expand(bn0,bn0)
          b_lifespans = lifespans.unsqueeze(1).expand(bn0,bn0)
          sum_lifespans = a_lifespans + b_lifespans
          
          wx_shifts = x_shifts * (b_lifespans/sum_lifespans)
          wy_shifts = y_shifts * (b_lifespans/sum_lifespans)
          
          # but for each a we need only do the largest x and largest y shift
          wx_shifts = torch.max(wx_shifts,dim = 1)[0] 
          wy_shifts = torch.max(wy_shifts,dim = 1)[0] 
          
          boxes[:,0] += wx_shifts * 0.5
          boxes[:,1] += wy_shifts * 0.5
          
          return boxes
          # for idx in range(bn0):
          #     id = ids[idx].item()
          #     tstate.kf.X[tstate.kf.obj_idxs[id],0] += wx_shifts[idx]
          #     tstate.kf.X[tstate.kf.obj_idxs[id],1] += wy_shifts[idx]