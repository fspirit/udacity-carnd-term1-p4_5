import itertools
import math
import cv2
import numpy as np

from scipy.ndimage import label
from features_extractor import FeaturesExtractor

class VehicleDetector:

    # scales = np.array([.5, .75, 1., 1.25, 1.5])
    # y_top = np.array([.55, .55, .56, .57, .57])
    # x_left = np.array([.5, .5, .45, .45, .45])
    # y_bottom = np.array([.85, .85, .85, .85, .9])

    scales = np.array([1., 1.25, 1.5, 2., 2.5])
    y_top = np.array([.55, .55, .56, .57, .57])
    x_left = np.array([.55, .55, .55, .55, .55])
    y_bottom = np.array([.85, .85, .85, .85, .9])

    def __init__(self, classifier, scaler, window_size=64, window_step=8, heatmap_threshold=1):
        self.classifier = classifier
        self.scaler = scaler
        self.window_size = window_size
        self.window_step = window_step
        self.heapmap_threshold = heatmap_threshold

    def num_of_steps(self, total_length):
        return ((total_length - self.window_size) // self.window_step) + 1

    def detect_vehicles(self, frame):

        detections = []
        (h, w, d) = frame.shape

        params = zip(VehicleDetector.y_top, VehicleDetector.x_left, VehicleDetector.y_bottom, VehicleDetector.scales)
        for y_top_min, x_left_min, y_bottom_max, scale in params:

            # Cut target area
            y_start = int(h * y_top_min)
            y_stop = int(h * y_bottom_max)
            x_start = int(w * x_left_min)

            frame_cropped = frame[y_start:y_stop, x_start:, :]

            frame_scaled = cv2.resize(frame_cropped, (np.int(frame_cropped.shape[1] / scale), np.int(frame_cropped.shape[0] / scale)))

            # Define number of step for x, y
            x_axis_steps = self.num_of_steps(frame_scaled.shape[1])
            y_axis_steps = self.num_of_steps(frame_scaled.shape[0])

            # Estimate HOGs for the whole image
            features_extractor = FeaturesExtractor(frame_scaled)

            # Enumerating windows
            for xw, yw in itertools.product(range(x_axis_steps), range(y_axis_steps)):

                # Get coords in pix
                x_left = xw * self.window_step
                y_top = yw * self.window_step

                # Getting path for current window
                patch = cv2.resize(frame_scaled[y_top:y_top+self.window_size, x_left:x_left+self.window_size],
                                   (self.window_size, self.window_size))

                features = features_extractor.extract(patch, x_left, y_top).reshape(1, -1)
                features = self.scaler.transform(features)

                if self.classifier.predict(features) == 1:

                    x_left_descaled = np.int(x_left * scale)
                    y_top_descaled = np.int(y_top * scale)
                    window_size_descaled = np.int(self.window_size * scale)

                    detections.append(
                        [(x_left_descaled + x_start, y_top_descaled + y_start),
                         (x_left_descaled + x_start + window_size_descaled , y_top_descaled + window_size_descaled + y_start)])

        return detections

    def get_heatmap(self, image, detections, threshold=1):

        heatmap = np.zeros((image.shape[0], image.shape[1])).astype(np.float)
        for box in detections:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        heatmap[heatmap <= threshold] = 0

        frame_with_heatmap = np.dstack((np.zeros((image.shape[0], image.shape[1])),
                                        np.zeros((image.shape[0], image.shape[1])),
                                        heatmap / np.max(heatmap) * 255))
        return heatmap, frame_with_heatmap

    def draw_boxes(self, original_img, heatmap):

        components = label(heatmap)
        image_with_components = np.dstack((components[0], components[0], components[0])) * 255

        image_with_boxes = np.copy(original_img)

        for car_number in range(1, components[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (components[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # Discard small bboxes
            (w, h) = (bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])
            size_threshold = self.window_size * 0.75
            if w <= size_threshold or h <= size_threshold:
                continue

            cv2.rectangle(image_with_boxes, bbox[0], bbox[1], (0, 0, 255), 6)

        return image_with_components, image_with_boxes

    def draw_all_detections(self, frame, detections):
        frame_with_detections = np.copy(frame)

        sizes = set([(d[0][0] - d[1][0], d[0][1] - d[1][1]) for d in detections])

        # Color windows with different scales to ease debugging
        size_to_color = {}
        for size in sizes:
            size_to_color[size] = np.random.randint(1, 6, size=3) * 50.0

        for detection in detections:
            color = size_to_color[(detection[0][0] - detection[1][0], detection[0][1] - detection[1][1])]

            cv2.rectangle(frame_with_detections,
                          (detection[0][0], detection[0][1]),
                          (detection[1][0], detection[1][1]),
                          color,
                          2)

        return frame_with_detections

    def run(self, frame):

        detections = self.detect_vehicles(frame)
        frame_with_detections = self.draw_all_detections(frame, detections)

        heatmap, frame_with_heatmap = self.get_heatmap(frame, detections, threshold=self.heapmap_threshold)
        frame_with_components, frame_with_final_boxes = self.draw_boxes(frame, heatmap)

        return frame_with_final_boxes, frame_with_components, frame_with_heatmap, frame_with_detections


class HistoryKeepingVehicleDetector(VehicleDetector):

    max_history_depth = 10

    def __init__(self, classifier, scaler, window_size=64, window_step=16, heatmap_threshold=2):
        super().__init__(classifier, scaler, window_size, window_step, heatmap_threshold)
        self.detections_history = []

    def run(self, frame):

        detections = self.detect_vehicles(frame)

        self.detections_history.insert(0, detections)

        if len(self.detections_history) > HistoryKeepingVehicleDetector.max_history_depth:
            self.detections_history.pop()

        non_empty_frames_count = sum([1 for detections in self.detections_history if len(detections) != 0])
        all_detections = list(itertools.chain(*self.detections_history))

        if len(all_detections) != 0:
            frame_with_detections = self.draw_all_detections(frame, all_detections)
            heatmap, frame_with_heatmap = \
                self.get_heatmap(frame,
                                 all_detections,
                                 max(self.heapmap_threshold,
                                     math.floor(non_empty_frames_count * self.heapmap_threshold * 0.5)))
            frame_with_components, result = self.draw_boxes(frame, heatmap)
        else:
            result = frame
            frame_with_detections = frame
            frame_with_components = np.zeros(frame.shape)
            frame_with_heatmap = np.zeros(frame.shape)

        # if len(detections)

        return result, frame_with_components, frame_with_heatmap, frame_with_detections
