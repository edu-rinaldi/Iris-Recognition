import argparse, sys
from ast import arg

import numpy as np
from scipy.spatial.distance import pdist

from Enrollment import Dataset, FeatureExtractor
from Models.VGGFE import VGGFE


class IdentityVerifier:
    def __init__(self, dataset : Dataset, acceptanceThreshold : float=2.3):
        self.dataset = dataset
        self.segmentator = dataset.segmentator
        self.featureExtractor = dataset.featureExtractor
        self.at = acceptanceThreshold

    def verify(self, img, claimedId : int) -> bool:
        _, segmented, _, _ = self.segmentator.segment(img, **self.dataset.launchParams)
        
        if type(segmented) == type(None):
            return False

        probeFeatures = self.featureExtractor.extract(segmented)
        df = self.dataset.dataframe()
        df = df[df['label'] == claimedId]
        # no user found with that claimed id
        if len(df) == 0:
            return False
        
        # calc distance with every template associated to that claimed id
        minDistance = sys.float_info.max
        for row in df.to_numpy():
            rowFeatures = row.reshape(1, -1)
            stacked = np.vstack((rowFeatures[:, :-1], probeFeatures.reshape(1, -1)))
            minDistance = min(minDistance, pdist(stacked, metric='euclidean')[0])
        
        return minDistance < self.at
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verification demo')
    parser.add_argument('--in', metavar='img_path', type=str, nargs=1, required=True,
                    help='Input image to verify')
    parser.add_argument('--id', metavar='claimed_identity', type=int, nargs=1, required=True,
                    help='Claimed identity')
    parser.add_argument('--dataset', metavar='dataset_filepath', type=str, nargs=1, required=True,
                    help='Dataset csv file')

    args = vars(parser.parse_args())
    inputImagePath = args['in'][0]
    claimedIdentity = args['id'][0]
    datasetPath = args['dataset'][0]
    vggfe = VGGFE(pretrainedName='vggfe_lr0001_100e.pth')
    featureExtractor = FeatureExtractor(vggfe)
    dataset = Dataset(datasetPath, featureExtractor)
    identityVerifier = IdentityVerifier(dataset)

    if identityVerifier.verify(inputImagePath, claimedIdentity):
        print(f"Identity verified, you are user#{claimedIdentity}")
    else:
        print("Access denied.")
