import numpy as np
import cv2


class PerspectiveTransformer(object):

    source_points = np.float32([[-100, 720], [555, 450], [725, 450], [1380, 720]])
    dist_points = np.float32([[200, 720], [200, 0], [1080, 0], [1080, 720]])


    # source = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
    # Define corresponding destination points
    # destination = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])

    def __init__(self):
        self.forward_transform_matrix = cv2.getPerspectiveTransform(PerspectiveTransformer.source_points,
                                                                    PerspectiveTransformer.dist_points)
        self.reverse_transform_matrix = cv2.getPerspectiveTransform(PerspectiveTransformer.dist_points,
                                                                    PerspectiveTransformer.source_points)

    def transform(self, frame):
        # Check image shape is ok
        return cv2.warpPerspective(frame, self.forward_transform_matrix, (frame.shape[1], frame.shape[0]))

    def reverse_transform(self, frame):
        return cv2.warpPerspective(frame, self.reverse_transform_matrix, (frame.shape[1], frame.shape[0]))