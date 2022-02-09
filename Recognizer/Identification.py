import argparse, sys
from pickle import FALSE
from math import dist
from ast import arg
from turtle import distance
from cv2 import distanceTransform
from importlib_metadata import List

import numpy as np
from scipy.spatial.distance import pdist

from Enrollment import Dataset, FeatureExtractor
from Models.VGGFE import VGGFE
from Models.FeatNetFE import FeatNet


class SubjectIdentifier:
    def __init__(self, dataset : Dataset, acceptanceThreshold : float=None):
        self.dataset = dataset
        self.segmentator = dataset.segmentator
        self.featureExtractor = dataset.featureExtractor

        self.at = acceptanceThreshold
    def identify(self, img, maxRank : int=1) -> List:
        _, segmented, _, _ = self.segmentator.segment(img, **self.dataset.launchParams)
        
        if type(segmented) == type(None):
            return []
        
        probeFeatures = self.featureExtractor.extract(segmented)
        df = self.dataset.dataframe()

        # no user found with that claimed id
        if len(df) == 0:
            return []

        # calc distance with every template
        distances = []
        for row in df.to_numpy():
            rowFeatures = row.reshape(1, -1)
            stacked = np.vstack((rowFeatures[:, :-1], probeFeatures.reshape(1, -1)))
            distances += [(pdist(stacked, metric='euclidean')[0], int(rowFeatures[0, -1]))]
        
        distances.sort(key=lambda el: el[0])
        if type(self.at) == type(float()):
            distances = list(filter(lambda x: x[0] < self.at, distances))
        return distances[:maxRank]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identification demo')
    parser.add_argument('--in', metavar='img_path', type=str, nargs=1, required=True,
                    help='Input image to identify')
    parser.add_argument('--at', metavar='th', type=float, nargs=1, required=False,
                    help='Acceptance theshold, if not set we assume it\'s a cloded set')
    parser.add_argument('--dataset', metavar='dataset_filepath', type=str, nargs=1, required=True,
                    help='Dataset csv file')

    args = vars(parser.parse_args())
    inputImagePath = args['in'][0]
    datasetPath = args['dataset'][0]
    if 'at' in args and args['at'] != None:
        at = args['at'][0]
    else:
        at = None
    
    # vggfe = VGGFE(pretrainedName='vggfe_lr0001_100e.pth')
    featNet = FeatNet(pretrainedName="featNetTriplet_100e_1e-4lr.pth").eval()

    featureExtractor = FeatureExtractor(featNet)
    dataset = Dataset(datasetPath, featureExtractor)
    subjectIdentifier = SubjectIdentifier(dataset, at)

    users = subjectIdentifier.identify(inputImagePath)
    if len(users) == 0:
        print(f"No match in the dataset")
    else:
        print(f"You are user#{users[0][1]}")
    
