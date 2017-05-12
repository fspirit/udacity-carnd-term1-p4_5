import glob
import cv2
import os

from sklearn.externals import joblib
from vehicle_detector import VehicleDetector
from frame_composer import FrameComposer

svc = joblib.load('svc_model.pkl')
scaler = joblib.load('scaler.pkl')

vd = VehicleDetector(svc, scaler)

files = glob.glob('./data/test_images/*.jpg')
# files = ['./data/test_images/test6.jpg']

for file in files:

    image = cv2.imread(file)
    final_boxes, components, heatmap, all_boxes = vd.run(image)

    for suffix, result_image in zip(('final', 'components', 'heatmap', 'all_boxes'),
                                    (final_boxes, components, heatmap, all_boxes)):
        base = os.path.basename(file)
        cv2.imwrite('./data/test_images_output/{}_{}.jpg'.format(os.path.splitext(base)[0], suffix), result_image)

        fc = FrameComposer(final_boxes)
        fc.add_upper_bar((components, heatmap, all_boxes))
        fc.add_text('Hey girl, hey boy!')

        cv2.imwrite('./data/test_images_output/{}_{}.jpg'.format(os.path.splitext(base)[0], 'mixed'), fc.get_frame())








