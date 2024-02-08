import torch
from torchvision.transforms import functional as F
from torchvision.ops import roi_align, nms
import numpy as np

def im_nms(detections,scores,threshold = 0.8,groups = None):
	"""
	Performs non-maximal supression on boxes given in image formulation
	detections - [d,8,2] array of boxes in state formulation
	scores - [d] array of box scores in range [0,1]
	threshold - float in range [0,1], boxes with IOU overlap > threshold are pruned
	groups - None or [d] tensor of unique group for each box, boxes from different groups will supress each other
	returns - idxs - list of indexes of boxes to keep
	"""

	minx = torch.min(detections[:,:,0],dim = 1)[0]
	miny = torch.min(detections[:,:,1],dim = 1)[0]
	maxx = torch.max(detections[:,:,0],dim = 1)[0]
	maxy = torch.max(detections[:,:,1],dim = 1)[0]

	boxes = torch.stack((minx,miny,maxx,maxy),dim = 1)

	if groups is not None:
	    large_offset = 10000
	    offset = groups.unsqueeze(1).repeat(1,4) * large_offset
	    boxes = boxes + large_offset

	idxs = nms(boxes,scores,threshold)
	return idxs

def space_nms(detections,scores,threshold = 0.1):
        """
        Performs non-maximal supression on boxes given in state formulation. Now 
        deprecated in favor of state_nms 
        detections - [d,8,3] array of boxes in  space formulation
        scores - [d] array of box scores in range [0,1]
        threshold - float in range [0,1], boxes with IOU overlap > threshold are pruned
        returns - idxs - indexes of boxes to keep
        """
        
        raise UserWarning("space_nms is deprecated. Use state_nms instead")
        
        # convert into xmin ymin xmax ymax form        
        boxes_new = torch.zeros([detections.shape[0],4],device = detections.device)
        boxes_new[:,0] = torch.min(detections[:,0:4,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(detections[:,0:4,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(detections[:,0:4,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(detections[:,0:4,1],dim = 1)[0]
                
        idxs = nms(boxes_new,scores,threshold)
        return idxs

def state_nms(boxes,scores,threshold  = 0.1):
    """
    Some fool designed space_nms assuming that the state axes would always be orthogonal to the space axes. I am that fool
    This is not the case. state_nms creates boxes expressed as 4 points in state coordinates rather than space coordinates
    so that nms can be performed without error. It will also serve as a drop-in replacement for the old space_nms
    
    boxes- [d,6] array of boxes in  state formulation
    scores - [d] array of box scores in range [0,1]
    threshold - float in range [0,1], boxes with IOU overlap > threshold are pruned
    returns - idxs - indexes of boxes to keep
    """

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
            
    idxs = nms(boxes_new,scores,threshold)
    return idxs



def md_iou(a,b):
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

def estimate_ts_offsets(boxes,camera_names,detection_times, tstate,hg, nms_space_thresh = 0.5,base_camera = "p1c1"):
     """
     Timestamps associated with each camera are assumed to have Gaussian error.
     The bias of this error is estimated as follows:
     On full frame detections, We find all sets of detection matchings across
     cameras. We estimate the expected time offset between the two based on 
     average object velocity in the same direction. We then 
     greedily solve the global time adjustment problem to minimize the 
     deviation between matched detections across cameras, after adjustment
     
     boxes            - [d,6] array of detected boxes in state form
     camera_names     - [d] array of camera indexes
     detection_times  - [d] list of times for each detection
     nms_space_thresh - float
     tstate           - trackState object because we need the filter for average speed
     hg               - Homography object
     """
     
     
     if len(boxes) == 0 or tstate.kf.X is None:
         return
     boxes = boxes.clone()
     
     ts_bias = dict([(key,0) for key in list(set(camera_names))])
     n_corrections = dict([(key,0) for key in list(set(camera_names))])
     
     EB_vel,WB_vel = tstate.kf.get_avg_speed()
     
     
     # boxes is [d,6] - need to convert into xmin ymin xmax ymax form
     boxes_space = hg.state_to_space(boxes)
     
     # convert into xmin ymin xmax ymax form        
     boxes_new =   torch.zeros([boxes_space.shape[0],4])
     boxes_new[:,0] = torch.min(boxes_space[:,0:4,0],dim = 1)[0]
     boxes_new[:,2] = torch.max(boxes_space[:,0:4,0],dim = 1)[0]
     boxes_new[:,1] = torch.min(boxes_space[:,0:4,1],dim = 1)[0]
     boxes_new[:,3] = torch.max(boxes_space[:,0:4,1],dim = 1)[0]
     
     # get iou for each pair
     dup1 = boxes_new.unsqueeze(0).repeat(boxes.shape[0],1,1).double()
     dup2 = boxes_new.unsqueeze(1).repeat(1,boxes.shape[0],1).double()
     iou = md_iou(dup1,dup2).reshape(boxes.shape[0],boxes.shape[0])
     
     # store offsets - offset is position in cam 2 relative to cam1
     x_offsets = []
     for i in range(iou.shape[0]):
         for j in range(i,iou.shape[1]):
             if i != j and camera_names[i] != camera_names[j]:
                 if iou[i,j] > nms_space_thresh:
                     x_offsets.append([camera_names[i],camera_names[j], boxes[j,0] - boxes[i,0],detection_times[i] - detection_times[j],boxes[i,5]])
                     x_offsets.append([camera_names[j],camera_names[i], boxes[i,0] - boxes[j,0],detection_times[j] - detection_times[i],boxes[i,5]])
    
     if len(x_offsets) == 0:
        return
     
     # Each x_offsets item is [camera 1, camera 2, x_offset, time_offset and direction]
     dx = torch.tensor([item[2] for item in x_offsets])
     dt_expected =  torch.tensor([item[3] for item in x_offsets]) # not 100% sure about the - sign here - logically I cannot work out its value
     
     #tensorize velocity
     vel = torch.ones(len(x_offsets)) * EB_vel
     for d_idx,item in enumerate(x_offsets):
         if item[3] == -1:
             vel[d_idx] = WB_vel
     
     # get observed time offset (x_offset / velocity)
     dt_obs = dx/vel
     time_error = dt_obs - dt_expected

     # each time_error corresponds to a camera pair
     # we could solve this as a linear program to minimize the total adjusted time_error
     # instead, we'll do a stochastic approximation
     
     # for each time error, we update self.ts_bias according to:
     # self.ts_bias[cam1] = (1-alpha)* self.ts_bias[cam1] + alpha* (-time_error + self.ts_bias[cam2])
     for e_idx,te in enumerate(time_error):
         cam1 = x_offsets[e_idx][0]
         cam2 = x_offsets[e_idx][1]
         if cam1 != base_camera: # by default we define all offsets relative to sequence 0
             ts_bias[cam1] = -te + ts_bias[cam2]
             n_corrections[cam1] += 1
     
     ts_offset = dict([(key,(ts_bias[key]/n_corrections[key]).item()) for key in ts_bias.keys() if n_corrections[key] > 0])
     #ts_offset = {key:val for key,val in ts_offset.items() if not np.isnan(val)}
     return ts_offset