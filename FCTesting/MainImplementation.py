import numpy as np
import pickle

from glob import glob

from SimpleCV import Image
from SimpleCV import HueHistogramFeatureExtractor
from SimpleCV import HaarLikeFeatureExtractor
from SimpleCV import EdgeHistogramFeatureExtractor

from matplotlib import pyplot as plt
import os

from .Services.MainServices import make_histogram

dataDir = os.path.dirname(os.path.abspath(__file__))
haar_file = os.path.join(dataDir, "../data/haar.txt")

haarfe = HaarLikeFeatureExtractor(fname=haar_file)
hhfe = HueHistogramFeatureExtractor(10)
ehfe = EdgeHistogramFeatureExtractor(10)

classifier = pickle.load(open(os.path.join(dataDir, "../data/classifier.pkl"), 'rb'))
labels = pickle.load(open(os.path.join(dataDir, "../data/featLabel.pkl"), 'rb'))


def predictor():
    for img in glob('/home/linuxsagar/tempTest/*'):

        #List to hold the feature of new Image
        _new_features = []

        #Read the test image
        _custom_image = Image(img)

        #Extract features
        _newHhfe = hhfe.extract(_custom_image)
        _newEhfe = ehfe.extract(_custom_image)
        _newHaarfe = haarfe.extract(_custom_image)


        #List for the graph
        hue_data = []
        haar_data = []
        edge_data = []

        hue_fields = hhfe.getFieldNames()
        for i in range(0, hhfe.getNumFields()):
            hue_data.append(str(hue_fields[i])+' = '+str(_newHhfe[i]))

        edge_fields = ehfe.getFieldNames()
        for i in range(0, hhfe.getNumFields()):
            edge_data.append(str(edge_fields[i])+' = '+str(_newEhfe[i]))

        haar_fields = haarfe.getFieldNames()
        for i in range(0, len(haar_fields)):
            haar_data.append(str(haar_fields[i])+' = '+str(_newHaarfe[i]))

        #Concatinate all feature to one
        _new_features.append(np.concatenate([_newHhfe, _newEhfe, _newHaarfe]))
        probability = np.amax(classifier.predict_proba(_new_features))

        #Call method to generate histogram image
        make_histogram(_custom_image.histogram(), _newHhfe, _newEhfe, _newHaarfe, dataDir)

        #The Final result to view
        result = ["The given input is classified as: {}.".format(classifier.predict(_new_features)[0]),
                      "Probability of prediction: {:.1%}".format((probability))]

        return [result,hue_data,edge_data,haar_data]

if __name__ == '__main__':
    predictor()