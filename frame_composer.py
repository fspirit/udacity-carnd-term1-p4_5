import cv2
import numpy as np


class FrameComposer(object):

    upper_bar_x_left = 20
    upper_bar_y_top = 20
    upper_bar_height = 250

    def __init__(self, base_frame):
        self.base_frame = base_frame
        self.has_upper_bar = False
        self.text_labels = []

    def add_upper_bar(self, components):
        if len(components) == 0:
            return

        # Darken upper part of base frame a bit
        self.base_frame[:FrameComposer.upper_bar_height, :, :] = self.base_frame[:FrameComposer.upper_bar_height, :, :] * .8

        components_ = components[:3]

        components_ = [cv2.resize(c.astype(np.float32), (0, 0), fx=0.3, fy=0.3) for c in components_]

        for i, c in enumerate(components_):
            (component_h, component_w, _) = c.shape
            y_from, y_to, x_from, x_to = (FrameComposer.upper_bar_y_top,
                                          FrameComposer.upper_bar_y_top + component_h,
                                          (FrameComposer.upper_bar_x_left + component_w) * i + FrameComposer.upper_bar_x_left,
                                          (FrameComposer.upper_bar_x_left + component_w) * (i + 1))

            self.base_frame[y_from:y_to, x_from:x_to, :] = c

        self.has_upper_bar = True

    def add_mask_over_base(self, mask):
        self.base_frame = cv2.addWeighted(self.base_frame, 1, mask, 0.3, 0)

    def add_text(self, text):
        y_top_offset, x_left_offset = (40, 40)

        if self.has_upper_bar:
            y_top_offset += FrameComposer.upper_bar_height

        y_top_offset += sum([60 for t in self.text_labels])

        cv2.putText(self.base_frame, text, (x_left_offset, y_top_offset), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 4)

        self.text_labels.append(text)

    def get_frame(self):
        return self.base_frame
