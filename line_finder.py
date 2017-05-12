from collections import deque

import cv2
import numpy as np


class LineFinder(object):

    windows_count = 9
    window_margin = 100
    window_minpix = 50

    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()

    def get_starting_x_bases(self, frame):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(frame[int(frame.shape[0] / 2):, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    def find_lines(self, frame_bin):

        frame_with_windows_n_lines = np.dstack((frame_bin, frame_bin, frame_bin)) * 255

        leftx_base, rightx_base = self.get_starting_x_bases(frame_bin)

        # Set height of windows
        window_height = np.int(frame_bin.shape[0] / LineFinder.windows_count)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = frame_bin.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(LineFinder.windows_count):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = frame_bin.shape[0] - (window + 1) * window_height
            win_y_high = frame_bin.shape[0] - window * window_height
            win_xleft_low = leftx_current - LineFinder.window_margin
            win_xleft_high = leftx_current + LineFinder.window_margin
            win_xright_low = rightx_current - LineFinder.window_margin
            win_xright_high = rightx_current + LineFinder.window_margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &
                               (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > LineFinder.window_minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > LineFinder.window_minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            cv2.rectangle(frame_with_windows_n_lines, (win_xleft_low, win_y_high), (win_xleft_high, win_y_low), (255, 255, 255), 4)
            cv2.rectangle(frame_with_windows_n_lines, (win_xright_low, win_y_high), (win_xright_high, win_y_low),
                      (255, 255, 255), 4)


        # Concatenate the arrays of indices
        left_lane_inds_cc = np.concatenate(left_lane_inds)
        right_lane_inds_cc = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx, lefty = nonzerox[left_lane_inds_cc], nonzeroy[left_lane_inds_cc]
        rightx, righty = nonzerox[right_lane_inds_cc], nonzeroy[right_lane_inds_cc]

        self.left_line.add_new_fit(leftx, lefty)
        self.right_line.add_new_fit(rightx, righty)

        pts_l = np.array(list(zip(*self.left_line.get_points())), np.int32).reshape((-1, 1, 2))
        pts_r = np.array(list(zip(*self.right_line.get_points())), np.int32).reshape((-1, 1, 2))

        cv2.polylines(frame_with_windows_n_lines, [pts_l, pts_r], False, (0, 255, 0), thickness=10)

        return frame_with_windows_n_lines, self.left_line, self.right_line


class Line(object):

    max_y = 720
    max_x = 1280

    ym_per_pix = 27 / 720
    xm_per_pix = 3.7 / 700

    def __init__(self):
        self.fits = deque(maxlen=5)

    def add_new_fit(self, x, y):
        enough_points = len(y) > 0 and np.percentile(y, 95) - np.percentile(y, 5) > Line.max_y * .6
        if enough_points or len(self.fits) == 0:
            self.fits.append(np.polyfit(y, x, 2))

    def averaged_fit(self):
        return np.array(self.fits).mean(axis=0)

    def get_points(self):
        y = np.linspace(0, self.max_y - 1, self.max_y)
        x = self.averaged_fit()[0] * y**2 + self.averaged_fit()[1] * y + self.averaged_fit()[2]
        return x, y

    def radius_of_curvature(self):
        x, y = self.get_points()
        fit_cr = np.polyfit(y * Line.ym_per_pix, x * Line.xm_per_pix, 2)
        return int(((1 + (2 * fit_cr[0] * 720 * Line.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0]))

    def camera_distance(self):
        x, y = self.get_points()
        x = x[np.argmax(y)]
        return np.absolute((self.max_x // 2 - x) * Line.xm_per_pix)