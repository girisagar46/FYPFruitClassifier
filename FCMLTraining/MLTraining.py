from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation

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
    features = list()
    labels = list()
    for im in images:
        try:
            img = Image(im)
            labels.append(im[:-2])
            features.append(np.concatenate([hhfe.extract(img),ehfe.extract(img),haarfe.extract(img)]))
        except:
            pass
    return[features, labels]


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
        scores = cross_validation.cross_val_score(clf, npa, npl, cv=10)
        print("n_estimators = 101 | learning_rate = {0} | cross_val_score = {1}".format(i, scores.mean()))


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
    pickle.dump(data, open('classifier.pkl', 'wb'))


# System testing
def test_system():

    # path of test folder
    for each in glob('./testing/*'):
        features2 = []
        print("Opening {}...".format(each))
        t_image = Image(each)
        features2.append(np.concatenate([hhfe.extract(t_image), ehfe.extract(t_image), haarfe.extract(t_image)]))
        t_data = pickle.load('classifier.pkl', 'rb')
        print(t_data.predict_proba(np.asarray(features2)))
        print(t_data.predict(np.asanyarray(features2)))

if __name__ == '__main__':
    training()
    #test_system()
