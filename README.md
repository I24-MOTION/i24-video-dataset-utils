
This repository contains utilities for working with the I-24 MOTION Video "I24V" dataset, which appeared at WACV 2024 in an article titled "So you think you can track?" (Gloudemans et al., 2024); the article is available here: https://openaccess.thecvf.com/content/WACV2024/papers/Gloudemans_So_You_Think_You_Can_Track_WACV_2024_paper.pdf. 

This is a multi-camera video dataset consisting of 234 hours of video data recorded concurrently from 234 overlapping HD cameras covering the 4.2 mile length of the I-24 MOTION testbed. The video is recorded during a period of high traffic density with 500+ objects typically visible within the scene and typical object longevities of 3-15 minutes. While dense object tracking information is infeasible to annotate, GPS trajectories from 270 vehicle passes through the scene are manually corrected in the video data to provide a set of ground-truth trajectories for recall-oriented tracking metrics, and object detections are provided for each camera in the scene (159 million total before cross-camera fusion).


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
       - dynamic/
       - reference/
    - final_detections.npy
    - final_gps.csv
    - final_manual.csv
    - final_redactions.cpkl
    - WACV2024_hg_save.cpkl



 # Code
