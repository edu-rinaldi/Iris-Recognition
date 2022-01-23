import sys, os, shutil
import argparse
from ast import arg


from Segmentation.Segmentation import Segmentator
from Models.VGGFE import VGGFE
from torchvision.io import read_image
from torchvision import transforms

import pandas as pd
import numpy as np


class FeatureExtractor:
    def __init__(self, model):
        self.model = model

        self.imgHeight = 64
        self.imgWidth = 200
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.imgHeight, self.imgWidth)),
            transforms.ToTensor()
        ])
    
    def extract(self, img):
        if type(img) == type(''):
            img = read_image(img)
        # prepare image for feature extraction
        img = self.transform(img)
        img = img.reshape(1, 3, self.imgHeight, self.imgWidth)

        return self.model(img)[0].detach().numpy()

class Dataset:
    def __init__(self, path, featureExtractor):
        self.path = path
        self.segmentator = Segmentator()
        self.launchParams = {"debug":True, "out":os.path.abspath("./.tmp"), "mode": "segmentation"}
        
        self.idCounter = 0
        
        self.featureExtractor = featureExtractor
        
        # private members
        self.__isCsvCreated = os.path.exists(self.path)

    def dataframe(self):
        if self.__isCsvCreated:
            return pd.read_csv(self.path)

    def enrollSubject(self, imgPath, id):
        # segment the image
        _, segmented, _, _ = self.segmentator.segment(imgPath, **self.launchParams)
        
        if type(segmented) == type(None):
            return False
        # extract features
        features = self.featureExtractor.extract(segmented) 
        
        # check if csv file is created, if not create it
        if not self.__isCsvCreated:
            df = pd.DataFrame([np.hstack((features, np.array([id])))], columns=['f'+str(i+1) for i in range(features.shape[0])]+["label"])
            self.__isCsvCreated = True
        else:
            df = self.dataframe()
            df = pd.concat([df, pd.DataFrame([np.hstack((features, np.array([id])))], columns=df.columns)])
        df.to_csv(self.path, index=False)
        return True

# Enrollment application
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Enrollment application')
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

    vggfe = VGGFE("vggfe_lr0001_100e.pth").eval()
    featureExtractor = FeatureExtractor(vggfe)
    dataset = Dataset(datasetPath, featureExtractor)
    dataset.enrollSubject(inputImagePath, claimedIdentity)