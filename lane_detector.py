import cv2
import numpy as np

from line_finder import LineFinder
from perspective_transformer import PerspectiveTransformer


class LaneDetector(object):

    def __init__(self, undistorter):
        self.undistorter = undistorter
        self.line_finder = LineFinder()

    def abs_sobel_thresh(self, gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
        sobel = np.zeros_like(gray)
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return grad_binary

    def mag_thresh(self, gray, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        # Take the absolute value of the derivative or gradient
        mag_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * mag_sobelxy / np.max(mag_sobelxy))

        mag_binary = np.zeros_like(scaled_sobel)
        mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

        return mag_binary

    def dir_threshold(self, gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate directional gradient
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        # Take the absolute value of the derivative or gradient
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        grad_direction = np.arctan2(abs_sobely, abs_sobelx)

        dir_binary = np.zeros_like(grad_direction)
        dir_binary[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1

        return dir_binary

    def apply_mask(self, frame_bin):
        (h, w) = frame_bin.shape
        vertices = np.array([(0, h), (w // 2 - 75, h * .625), (w // 2 + 75, h * .625), (w, h)],  dtype=np.int32)
        mask = np.zeros_like(frame_bin)
        mask_color = 1

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillConvexPoly(mask, vertices, mask_color)

        masked_image = cv2.bitwise_and(frame_bin, mask)
        return masked_image

    def apply_grad_and_color_thresholds(self, frame):
        # Choose a Sobel kernel size
        ksize = 15  # Choose a larger odd number to smooth gradient measurements

        frame_bin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply each of the thresholding functions
        gradx_bin = self.abs_sobel_thresh(frame_bin, orient='x', sobel_kernel=ksize, thresh=(20, 100))
        grady_bin = self.abs_sobel_thresh(frame_bin, orient='y', sobel_kernel=ksize, thresh=(20, 100))
        grad_mag_bin = self.mag_thresh(frame_bin, sobel_kernel=ksize, mag_thresh=(20, 100))
        grad_dir_bin = self.dir_threshold(frame_bin, sobel_kernel=ksize, thresh=(0.7, 1.3))

        hls_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        s_image = hls_image[:, :, 2]

        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_image = hsv_image[:, :, 2]

        combined = np.zeros_like(grad_dir_bin)

        #     combined[(canny_result == 1) |
        #             ((gradx == 1) & (grady == 1)) |
        #              ((mag_binary == 1) & (dir_binary == 1)) |
        #              ((s_image >= 85) & (s_image <= 190)) |
        #             ((v_image >= 234) & (v_image <= 255))] = 1

        combined[((gradx_bin == 1) & (grady_bin == 1)) |
                 ((s_image >= 85) & (s_image <= 190)) |
                 ((v_image >= 230) & (v_image <= 255))] = 1

        #     combined[
        #          ((gradx == 1) & (grady == 1)) |
        #             (mag_binary == 1) |
        #          ((s_image >= 170) & (s_image <= 255)) &
        #         ((v_image >= 234) & (v_image <= 255))  ] = 1

        combined_n_masked = self.apply_mask(combined)

        return combined_n_masked, np.dstack((combined_n_masked, combined_n_masked, combined_n_masked)) * 255

    def get_empty_frame_with_lane(self, frame, left_line, right_line):
        # Create an image to draw the lines on
        frame_with_lane = np.zeros_like(frame).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_l = np.array([np.transpose(np.vstack(left_line.get_points()))])
        pts_r = np.array([np.flipud(np.transpose(np.vstack(right_line.get_points())))])
        pts = np.hstack((pts_l, pts_r))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(frame_with_lane, np.int_([pts]), (0, 255, 0))

        return frame_with_lane

    def run(self, frame):

        undistorted_frame = self.undistorter.undistort(frame)

        thresholded_frame_bin, thresholded_frame = self.apply_grad_and_color_thresholds(undistorted_frame)

        pt = PerspectiveTransformer()

        frame_2d_bin = pt.transform(thresholded_frame_bin)

        frame_with_lines_n_windows_2d, left_line, right_line = self.line_finder.find_lines(frame_2d_bin)

        frame_with_lane_filled_2d = self.get_empty_frame_with_lane(frame, left_line, right_line)

        frame_with_lane_filled = pt.reverse_transform(frame_with_lane_filled_2d)

        radius_of_curvature = int(np.mean([left_line.radius_of_curvature(), right_line.radius_of_curvature()]))

        return frame_with_lane_filled, frame_with_lines_n_windows_2d, thresholded_frame, radius_of_curvature
