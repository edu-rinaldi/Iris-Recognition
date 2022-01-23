import sys, os, shutil


from Segmentation.Segmentation import Segmentator
from Models.VGGFE import VGGFE
from torchvision.io import read_image
from torchvision import transforms

import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, path, featureExtractor):
        self.path = path
        self.segmentator = Segmentator()
        self.launchParams = {"debug":True, "out":os.path.abspath("./.tmp"), "mode": "segmentation"}
        
        self.imgHeight = 64
        self.imgWidth = 200
        self.idCounter = 0
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.imgHeight, self.imgWidth)),
            transforms.ToTensor()
        ])
        
        self.featureExtractor = featureExtractor
        
        # private members
        self.__isCsvCreated = os.path.exists(self.path)

    def enrollSubject(self, imgPath, id):
        # segment the image
        self.segmentator.segment(imgPath, **self.launchParams)
        
        # save it temp.
        imgName = os.path.basename(imgPath)
        extIdx = imgName.rfind(".")
        imgExt = imgName[extIdx:]
        imgName = imgName[:extIdx]
        segmentedPath = os.path.join(".tmp", imgName+"_eyeNorm" + imgExt)
        # load it
        img = read_image(segmentedPath)
        # delete temp. files
        shutil.rmtree(os.path.dirname(segmentedPath))
        
        # prepare image for feature extraction
        img = self.transform(img)
        img = img.reshape(1, 3, self.imgHeight, self.imgWidth)

        # extract features
        features = self.featureExtractor(img)[0].detach().numpy() # first item of batch

        # check if csv file is created, if not create it
        if not self.__isCsvCreated:
            df = pd.DataFrame([np.hstack((features, np.array([id])))], columns=['f'+str(i+1) for i in range(features.shape[0])]+["label"])
            self.__isCsvCreated = True
        else:
            df = pd.read_csv(self.path)
            # print(pd.DataFrame([np.hstack((features, np.array([id])))], columns=df.columns))
            df = pd.concat([df, pd.DataFrame([np.hstack((features, np.array([id])))], columns=df.columns)])
        df.to_csv(self.path, index=False)

# test
if __name__ == "__main__":
    vggfe = VGGFE("vggfe_lr0001_100e.pth").eval()
    dataset = Dataset("./prova.csv", vggfe)
    dataset.enrollSubject(sys.argv[1], sys.argv[2])