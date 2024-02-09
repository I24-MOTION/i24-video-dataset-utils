
This repository contains utilities for working with the I-24 MOTION Video "I24V" dataset, which appeared at WACV 2024 in an article titled "So you think you can track?" (Gloudemans et al., 2024); the article is available [here](https://openaccess.thecvf.com/content/WACV2024/papers/Gloudemans_So_You_Think_You_Can_Track_WACV_2024_paper.pdf).

This is a multi-camera video dataset consisting of 234 hours of video data recorded concurrently from 234 overlapping HD cameras covering the 4.2 mile length of the I-24 MOTION testbed. The video is recorded during a period of high traffic density with 500+ objects typically visible within the scene and typical object longevities of 3-15 minutes. While dense object tracking information is infeasible to annotate, GPS trajectories from 270 vehicle passes through the scene are manually corrected in the video data to provide a set of ground-truth trajectories for recall-oriented tracking metrics, and object detections are provided for each camera in the scene (159 million total before cross-camera fusion).


![Example frame](readme_im/detections_and_gps.png)
*Example frames with GPS-annotated vehicle positions (green) and detection set (white) shown*

# Data Use Agreement
1. You are free to use the data in academic and commercial work. 
2. The dataset contains images comprised of anonymous vehicles. Any activities to re-identify individuals in the dataset or activities that may cause harm to individuals in the dataset are prohibited.
3. When you use the data or other I-24 MOTION data in published academic work, you are required to include the following citation contents for this dataset and the I-24 MOTION system. This allows us to aggregate statistics on the data use in publications:
   
**I24V Dataset**:

    @inproceedings{gloudemans2024so,
      title={So you think you can track?},
      author={Gloudemans, Derek and Zach{\'a}r, Gergely and Wang, Yanbing and Ji, Junyi and Nice, Matt and Bunting, Matt and Barbour, William W and Sprinkle, Jonathan and Piccoli, Benedetto and Monache, Maria Laura Delle and others},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
      pages={4528--4538},
      year={2024}
    }
    
**I24-MOTION System**:

    @article{gloudemans202324,
      title={I-24 MOTION: An instrument for freeway traffic science},
      author={Gloudemans, Derek and Wang, Yanbing and Ji, Junyi and Zachar, Gergely and Barbour, William and Hall, Eric and Cebelak, Meredith and Smith, Lee and Work, Daniel B},
      journal={Transportation Research Part C: Emerging Technologies},
      volume={155},
      pages={104311},
      year={2023},
      publisher={Elsevier}
    }

4. You are free to create and share derivative products as long as you maintain the terms above.
5. The data is provided “As is.” We make no other warranties, express or implied, and hereby disclaim all implied warranties, including any warranty of merchantability and warranty of fitness for a particular purpose.

# Data 

The following files are included in this dataset:
- video/
    - PXXCXX_<ts>.mkv  - a single 1080p video file for the camera XX (01-06) on pole XX (01-40) (see MOTION system paper for details). Camera frames are encoded with millesecond-accurate timestamps, with the first frame timestamp being 0.000s for all videos. The <ts> included in the file name is an offset (in nanoseconds) which must be added to the encoded timestamps for a sequence to synchronize times globally across all videos.
    ... for each camera (234 total)
- annotations/
    - reference/ 
       - 4k/
          - PXXCXX.png - a 4k-resolution averaged frame from the specified camera, roughly showing the background without vehicles for this camera. These frames were used to align the homographies for each camera
          ... for each camera (234 total)
       - 1080p/
          - PXXCXX.png - a 1080p-resolution averaged frame from the specified camera, roughly showing the background without vehicles for this camera. These have the same resolution as the included video sequence
          ... for each camera (234 total)
    - track/
       - <tracking_run_name>_results.cpkl - stored results from running the specified tracking-by-detection algorithm on the included detection set. File is a pickle file consisting of:
              list of (tensor,int) - each tensor corresponds to a single tracked object and is of size [n_positions,7], with each row consisting of: time,x_pos,y_pos,length,width,height,<garbage>,detection_confidence. The int in the tuple indicates the vehicle class.
       ... for all tracking runs
       - results_ORACLE.gps  - stored results from running the oracle tracker on the included detection set. File is a pickle file consisting of:
              dictionary of tensors keyed by corresponding ground-truth object ID - each tensor corresponds to a single tracked object and is of size [n_positions,7], with each row consisting of: time,x_pos,y_pos,length,width,height,<garbage>,detection_confidence. 
       - GPS.cpkl - stores GPS positional data (the same data as in final_gps.csv but raveled by object id rather than observation to save time for tracking evaluation).
              dictionary of dictionaries keyed by object ID - each dictionary has fields:

                   id - unique object id (int)
                   run - pass-number for this vehicle through the camera system (int)
                   x  - tensor of x_positions (ft)
                   y - tensor of y_positions (ft)
                   ts - tensor of timestamps
                   l - object length (ft)
                   w - object width (ft)
                   h - object height (ft)
       
    - hg/
       - static/
           - H_PXXCXX_WB.npy - 3x3 matrix of H (used for image-space 2D point transformation) for the westbound side of the roadway in this camera based on the best-fit average homography over the 1 hour of video recording
           - H_PXXCXX_EB.npy - 3x3 matrix of H (used for image-space 2D point transformation) for the eastbound side of the roadway in this camera ...
           - P_PXXCXX_WB.npy - 3x4 matrix of P (used for space-image 3D point transformation) for the westbound side of the roadway in this camera ...
           - P_PXXCXX_EB.npy - 3x4 matrix of P (used for space-image 3D point transformation) for the eastbound side of the roadway in this camera ...
           ... for each camera where a valid homography can be defined (otherwise array will be filled with nans)
       - dynamic/
           - H_PXXCXX_WB.npy - Nx3x3 matrix of H (used for image-space 2D point transformation) for the westbound side of the roadway in this camera. Each item along the 0th dimension corresponds to the homography matrix for a single point in time (computed at 10-second intervals. The correct time-varying homography can be determined by dividing the timestamp by this 10-second interval and selecting the closest-indexed matrix).
           - H_PXXCXX_EB.npy - Nx3x3 matrix of H (used for image-space 2D point transformation) for the eastbound side of the roadway in this camera. ...
           - P_PXXCXX_WB.npy - Nx3x4 matrix of P (used for space-image 3D point transformation) for the westbound side of the roadway in this camera. ...
           - P_PXXCXX_EB.npy - Nx3x4 matrix of P (used for space-image 3D point transformation) for the eastbound side of the roadway in this camera. ...
           ... for each camera where a valid homography can be defined (otherwise array will be filled with nans)
       - reference/ 
           - H_PXXCXX_WB.npy - same format as static/ but each homography is uncorrected (i.e. relative to the original reference image in reference/)
           - H_PXXCXX_EB.npy - ...
           - P_PXXCXX_WB.npy - ...
           - P_PXXCXX_EB.npy - ...
           ... for each camera where a valid homography can be defined (otherwise array will be filled with nans)
    
    - final_detections.npy - numpy array of size [n_total_detections,8] with each row corresponding to: ["time (s)","Roadway X (ft)","Roadway Y (ft)","length (ft)","width (ft)","height (ft)","class","det confidence"]
    - final_gps.csv        - csv file with each row corresponding to a gps recorded (and subsequently corrected) vehicle position: ['id', 'State Plane X (ft)', 'State Plane Y (ft)', 'Roadway X (ft)', 'Roadway Y (ft)', 'Timestamp (s)', 'Length (ft)', 'Width (ft)', 'Height (ft)']
    - final_manual.csv     - csv file with each row corresponding to a manually annotated vehicle position: ['id', 'Roadway X (ft)', 'Roadway Y (ft)', 'Timestamp (s)', 'Length (ft)', 'Width (ft)', 'Height (ft)']
    - final_redactions.cpkl - a pickle file containing a dictionary keyed by camera name (e.g. "P09C04"). Each entry contains a list of redacted polygon-regions with a start and end time such that the user can deal with these regions as desired (e.g. black pixels or random noise).
    - WACV2024_hg_save.cpkl - a saved file containing all relevant homography and coordinate system data to initialize an `I24_RCS object` (see below) without a need for additional data files.



 # Code
A few minimal scripts are included to get you started working with the data:

## Requirements:
   - i24_rcs - implements the coordinate system and homography containers for easy coordinate system transformations. Available [here](https://github.com/DerekGloudemans/i24_rcs).
   - py-motmetrics - needed for evaluation of tracking results. Available [here](https://github.com/cheind/py-motmetrics)
   - Nvidia VPF - needed for loading video frames and timestamps using hardware acceleration and python bindings (you can work with the data without this but given the file size it will be tremendously slow and burdensome). Available [here](https://github.com/NVIDIA/VideoProcessingFramework).


## data_viewer.py Usage

    from data_viewer import DataViewer
    
    # specify inputs
    gps_path       = "<local dataset path>/final_gps.csv"  
    manual_path    = "<local dataset path>/final_manual.csv" 
    detection_path = "<local dataset path>/final_detections.npy" 
    video_dir      = "<local dataset path>/video" 
    hg_path        = "<local dataset path>/WACV2024_hg_save.cpkl" 
    
    camera_names   = ["P20C01","P20C02","P20C03","P20C04","P20C05","P20C06"] # for example, show 6 cameras - loading time and memory usage scales linearly with # of cameras
    buffer_window  = 4000 # number of frames to buffer starting with specified time
    start_time     = 200   # time in seconds from synchronized start of recording period (first frame timestamp is 0 roughly)
    
    # initialize DataViewer object
    dv = DataViewer(video_dir,
                    camera_names,
                    hg_path,
                    buffer_frames = buffer_window,
                    start_time = start_time, 
                    gps = gps_path,
                    manual = manual_path,
                    detections = detection_path)
   
    dv.run()

![](readme_im/ex.webm)

## evaluate.py Usage
For comparing the results of a tracking run against ground-truth trajectories (from GPS data)

    from evaluate import evaluate

    # specify inputs
    gps_path = "<local dataset path>/track/GPS.cpkl"
    track_path = "<local dataset path>/track/results_KIOU_10Hz.cpkl" # for example
    eval_stride = 0.1 # interpolate GPS and tracked vehicle positions at (say) 0.1sec intervals
    iou_threshold = 0.1 # required iou for a considered match between ground truth and predicted vehicle position
    
    evaluate(gps_path,track_path,eval_stride,iou_threshold)
  
            
