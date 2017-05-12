import argparse
import os
import pickle

from moviepy.editor import VideoFileClip
from vehicle_detector import HistoryKeepingVehicleDetector
from sklearn.externals import joblib
from frame_composer import FrameComposer
from lane_detector import LaneDetector
from undistorter import Undistorter

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Detect cars on video')
    parser.add_argument('video_file_path')

    args = parser.parse_args()

    original_video = VideoFileClip(args.video_file_path)

    svc = joblib.load('svc_model.pkl')
    scaler = joblib.load('scaler.pkl')

    with open('./undistorter.pkl', 'rb') as f:
        undistorter = pickle.load(f)

    vd = HistoryKeepingVehicleDetector(svc, scaler)
    ld = LaneDetector(undistorter)

    def process_frame(frame):
        final_boxes, components, heatmap, all_boxes = vd.run(frame)
        frame_with_lane_filled, frame_with_lines_n_windows_2d, thresholded_frame, radius_of_curvature = ld.run(frame)

        # fc = FrameComposer(final_boxes)
        # fc.add_mask_over_base(frame_with_lane_filled)
        # fc.add_upper_bar((thresholded_frame, frame_with_lines_n_windows_2d, all_boxes))
        # fc.add_text('Radius of curvature: {}'.format(radius_of_curvature))

        fc = FrameComposer(all_boxes)
        fc.add_mask_over_base(frame_with_lane_filled)
        fc.add_upper_bar((thresholded_frame, frame_with_lines_n_windows_2d, final_boxes))
        fc.add_text('Radius of curvature: {}'.format(radius_of_curvature))

        return fc.get_frame()

    output_video = original_video.fl_image(lambda frame: process_frame(frame))

    base = os.path.basename(args.video_file_path)
    output_video.write_videofile('./data/video_output/{}_out.mp4'.format(os.path.splitext(base)[0]), audio=False)

