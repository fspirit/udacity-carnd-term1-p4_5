import glob
import cv2
import os
import pickle

from lane_detector import LaneDetector
from frame_composer import FrameComposer
from undistorter import Undistorter

with open('./undistorter.pkl', 'rb') as f:
    undistorter = pickle.load(f)


ld = LaneDetector(undistorter)

files = glob.glob('./data/test_images/*.jpg')
# files = ['./data/test_images/test1.jpg']

for file in files:

    image = cv2.imread(file)
    frame_with_lane_filled, frame_with_lines_n_windows_2d, thresholded_frame, radius_of_curvature = ld.run(image)

    base = os.path.basename(file)

    fc = FrameComposer(image)
    fc.add_mask_over_base(frame_with_lane_filled)
    fc.add_upper_bar((frame_with_lines_n_windows_2d, thresholded_frame))
    fc.add_text('Raduis of curvature: {}'.format(radius_of_curvature))

    cv2.imwrite('./data/test_images_output/{}_{}.jpg'.format(os.path.splitext(base)[0], 'lane_detected'), fc.get_frame())