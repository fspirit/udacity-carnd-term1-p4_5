import glob

import numpy as np
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from features_extractor import FeaturesExtractor
from sklearn.svm import LinearSVC
from sklearn.externals import joblib


def features_from_files(files):

    features = []
    for file in files:
        image = cv2.imread(file)

        feature_extractor = FeaturesExtractor(image)
        features.append(feature_extractor.extract(image))

    return np.array(features)

if __name__ == "__main__":

    print("Starting to extract features")

    cars = glob.glob('./data/vehicles/**/*.png')
    nocars = glob.glob('./data/non-vehicles/**/*.png')

    car_features = features_from_files(cars)
    nocar_features = features_from_files(nocars)

    print("Features extracted, starting to train model")

    X = np.vstack((car_features, nocar_features)).astype(np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(nocar_features))))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=76676)

    svc = LinearSVC()
    svc.fit(X_train, y_train)

    print("Model score {:.4f}".format(svc.score(X_test, y_test)))

    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(svc, 'svc_model.pkl')
