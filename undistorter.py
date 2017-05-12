import numpy as np
import glob
import cv2
import pickle


class Undistorter(object):

    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)

    def calibrate(self):

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d points in real world space
        img_points = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('./data/camera_cal/calibration*.jpg')

        img_h, img_w = (None, None)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img_h is None or img_w is None:
                img_h, img_w = (gray.shape[1], gray.shape[0])

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret:
                obj_points.append(objp)
                img_points.append(corners)

        ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (img_h, img_w), None, None)

        if ret:
            self.camera_matrix = mtx
            self.dist_coeffs = dist

    def undistort(self, frame):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return None

        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)


if __name__ == "__main__":
    ud = Undistorter()
    ud.calibrate()

    img = cv2.imread('./data/camera_cal/calibration1.jpg')
    undistorted_img = ud.undistort(img)

    cv2.imwrite('./data/undistorter_test/calibration1_undistorted.jpg', undistorted_img)

    with open('undistorter.pkl', 'wb') as f:
        pickle.dump(ud, f)
