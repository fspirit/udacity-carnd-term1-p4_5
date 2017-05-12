My code for projects 4 and 5 in Udacity SDC Nanodegree
====

The tasks were
- Identify and highlight a road lane, on which the car with the camera is driving 
- Detect and highlight (with bounding boxes) other vehicles on the road

Lane detection
==

To find and identify the lines to the left and to the right of the car, the following actions were taken
- The frame was undistorted to recover original distances and sizes (see [undistorter.py](./undistorter.py))
- Lines and edges in the frame were detected, using thresholds for gradients and color components from different color spaces
(see [various instance methods in lane_detector](./lane_detector.py))
- The perspective of the frame was transformed to the birds' eye view, 
to get a snapshot of a road from above (see [perspective_transformer.py](./perspective_transformer.py)) 
- Sliding windows from bottom to the top of a frame were employed to find left and 
right lines (see [line_finder.py](./line_finder.py))
- Finally a frame was transformed back to original perspective and a detected lane was drawn on top of an original frame

To tackle the changes in lighting of the scene and shadows from off-road objects, an average of a number of previously detected lines 
was used and bad frames (very few non-null points) were filtered out. (see [Line](/.line_finder.py))  

Vehicle detection and tracking
==

To detect other vehicles on the road, the following steps were taken:
- An SVM classifier was trained on the dataset, which had been provided by Udacity ([train_svm_model.py](./train_svm_model.py))
Features used were: color bins, HOGs (see [features_extractor.py](./features_extractor.py))
- Sliding windows of several different sizes were run on top of the frame, each window was marked 1 if a classifier was positive,
 otherwise - 0. Each size was ran on a suitable part of a frame, big cars can only be found at the bottom of a frame, 
 little cars - closer to the middle, and we were only interested in the vehicles driving the same direction, 
 thus in the right part of the frame (see [detect_vehicles in vehicle_detector](./vehicle_detector.py)). 
- After a heatmap (+1 for any pixel that is inside of a window that was marked as detection) of all detections was created 
and a threshold applied (see [get_heatmap in vehicle_detector](./vehicle_detector.py)), 
a `label` function from `scipy.ndimage` was used to find connected components (whole patches of white)
- Connected components were bounded by boxes
- The resulting boxes were drawn on top of the original frame

False positives and false negatives were handled again by keeping history of detections
and taking in consideration all of those detections at once.

An awesome result
==

<a href="http://www.youtube.com/watch?feature=player_embedded&v=YOUTUBE_VIDEO_ID_HERE
" target="_blank"><img src="http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>


Things to improve
==

- The resulting classifier is not very good (a lot of false positives) and can be improved by getting more data from
 public datasets and by employing a different model (CNN) 
- Some more tricks can be used to improve lines and edges detection and preventing 
shadows to influence the results (better usage of color spaces, using history for sliding windows)
