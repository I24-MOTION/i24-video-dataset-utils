
This repository contains utilities for working with the I-24 MOTION Video "I24V" dataset, which appeared at WACV 2024 in an article titled "So you think you can track?" (Gloudemans et al., 2024); the article is available here: https://openaccess.thecvf.com/content/WACV2024/papers/Gloudemans_So_You_Think_You_Can_Track_WACV_2024_paper.pdf. 

This is a multi-camera video dataset consisting of 234 hours of video data recorded concurrently from 234 overlapping HD cameras covering the 4.2 mile length of the I-24 MOTION testbed. The video is recorded during a period of high traffic density with 500+ objects typically visible within the scene and typical object longevities of 3-15 minutes. While dense object tracking information is infeasible to annotate, GPS trajectories from 270 vehicle passes through the scene are manually corrected in the video data to provide a set of ground-truth trajectories for recall-oriented tracking metrics, and object detections are provided for each camera in the scene (159 million total before cross-camera fusion).


# Data Use Agreement
1. You are free to use the data in academic and commercial work. 
2. The dataset contains images comprised of anonymous vehicles. Any activities to re-identify individuals in the dataset or activities that may cause harm to individuals in the dataset are prohibited.
3. When you use the data or other I-24 MOTION data in published academic work, you are required to include the following citation contents for this dataset and the I-24 MOTION system. This allows us to aggregate statistics on the data use in publications:
   
I24V Dataset:

    @inproceedings{gloudemans2024so,
      title={So you think you can track?},
      author={Gloudemans, Derek and Zach{\'a}r, Gergely and Wang, Yanbing and Ji, Junyi and Nice, Matt and Bunting, Matt and Barbour, William W and Sprinkle, Jonathan and Piccoli, Benedetto and Monache, Maria Laura Delle and others},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
      pages={4528--4538},
      year={2024}
    }
    
I24-MOTION System:

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
