from skimage.feature import hog
import numpy as np
import cv2


class FeaturesExtractor(object):

    def __init__(self, image, color=cv2.COLOR_BGR2HSV, orient=10, pix_per_cell=8, cells_per_block=2):
        self.pix_per_cell = pix_per_cell

        image = cv2.cvtColor(image, color)

        hogs_ = []
        for ch in range(image.shape[2]):
            hogs_.append(
                hog(image[:, :, ch],
                    orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cells_per_block, cells_per_block),
                    transform_sqrt=True,
                    feature_vector=False))

        self.hogs = np.array(hogs_)

    def hog(self, patch, x, y):

        x_cell = x // self.pix_per_cell
        y_cell = y // self.pix_per_cell

        cells_per_patch = patch.shape[0] // self.pix_per_cell - 1

        patch_hogs = [hog[y_cell:y_cell + cells_per_patch, x_cell:x_cell + cells_per_patch].ravel() for hog in self.hogs]

        return np.hstack(patch_hogs)

    def bin_spatial(self, image, size=(16, 16)):

        return cv2.resize(image, size).ravel()

    def color_hist(self, image, nbins=16, bins_range=(0, 256)):

        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)

        return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    def extract(self, image, x=0, y=0):
        features = []

        spatial_features = self.bin_spatial(image)
        features.append(spatial_features)

        hist_features = self.color_hist(image)
        features.append(hist_features)

        hog_features = self.hog(image, x, y)
        features.append(hog_features)

        return np.concatenate(features)

