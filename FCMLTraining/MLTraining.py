import contextlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

from SimpleCV import Image
from SimpleCV import HueHistogramFeatureExtractor
from SimpleCV import HaarLikeFeatureExtractor
from SimpleCV import EdgeHistogramFeatureExtractor

import numpy as np
from glob import glob
import pickle

# model to extract feature
hhfe = HueHistogramFeatureExtractor(10)
ehfe = EdgeHistogramFeatureExtractor(10)
haarfe = HaarLikeFeatureExtractor('haar.txt')

# Give path of the training folder
images = glob('./fruits/*')


# Extract features and target labels for training
def get_feature_labels():
    features = []
    labels = []
    for im in images:
        with contextlib.suppress(Exception):
            img = Image(im)
            labels.append(im[:-2])
            features.append(
                np.concatenate([hhfe.extract(img), ehfe.extract(img), haarfe.extract(img)]))
    return [features, labels]


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


feat_label = get_feature_labels()


# Method to choose best learning rate
def choose_learning_rate():
    for i in frange(0.1, 1.0, 0.1):
        rf = AdaBoostClassifier(n_estimators=100, learning_rate=i)
        clf = Pipeline(steps=[('preprocess', StandardScaler()), ('clasifier', rf)])
        feat = feat_label[0]
        label = feat_label[1]
        npa = np.asarray(feat)
        npl = np.asarray(label)
        scores = cross_val_score(clf, npa, npl, cv=10)
        print(f"n_estimators = 101 | learning_rate = {i} | cross_val_score = {scores.mean()}")


# Training function
def training():
    pickle.dump(feat_label, open('featLabel.pkl', 'wb'))
    adb = AdaBoostClassifier(n_estimators=100, learning_rate=0.7)
    clf = Pipeline(steps=[('preprocess', StandardScaler()), ('clasifier', adb)])
    feat = feat_label[0]
    label = feat_label[1]
    npa = np.asarray(feat)
    npl = np.asarray(label)
    data = clf.fit(npa, npl)
    with open("classifier.pkl", "wb") as file:
        pickle.dump(data, file)


# System testing
def test_system():
    # path of test folder
    with open("classifier.pkl", "rb") as file:
        for each in glob('./testing/*'):
            print(f"Opening {each}...")
            t_image = Image(each)
            features2 = [
                np.concatenate(
                    [
                        hhfe.extract(t_image),
                        ehfe.extract(t_image),
                        haarfe.extract(t_image),
                    ]
                )
            ]
            t_data = pickle.load(file)
            print(t_data.predict_proba(np.asarray(features2)))
            print(t_data.predict(np.asanyarray(features2)))


if __name__ == '__main__':
    training()
    # test_system()
