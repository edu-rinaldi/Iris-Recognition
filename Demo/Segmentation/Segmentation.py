import sys
import os
import subprocess
import shutil

# from torchvision.io import read_image
import cv2

def read_image(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class Segmentator:
    def __init__(self):
        self.path = "./bin/SegmentatorApp"

    def segment(self, imagePath, **kwargs):
        imagePath = os.path.abspath(imagePath)
        listCommand = [self.path, "--in", imagePath]
        flagMode = False
        if 'method' in kwargs:
            listCommand += ["--method", kwargs["method"]]
        if 'size' in kwargs:
            listCommand += ["--size", kwargs["size"]]
        if 'mode' in kwargs:
            listCommand += ["--mode", kwargs["mode"]]
            flagMode = kwargs["mode"] == "segmentation"
        if 'out' in kwargs and flagMode:
            listCommand += ["--out", kwargs["out"]]
        if 'debug' in kwargs and kwargs["debug"]:
            command = " ".join(listCommand)
            print(f"Launching command: {command}")
        origWD = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.Popen(listCommand)
        proc.wait()
        os.chdir(origWD)

        if kwargs['mode'] == 'debug':
            return None, None, None, None
        # save it temp.
        imgName = os.path.basename(imagePath)
        extIdx = imgName.rfind(".")
        imgExt = imgName[extIdx:]
        imgName = imgName[:extIdx]
        outPath = kwargs['out']
        eyeNormPath = os.path.join(outPath, imgName+"_eyeNorm" + imgExt)
        eyePath = os.path.join(outPath, imgName+"_eye" + imgExt)
        eyeNormMaskPath = os.path.join(outPath, imgName+"_eyeNormMask" + imgExt)
        eyeMaskPath = os.path.join(outPath, imgName+"_eyeMask" + imgExt)

        if not os.path.exists(eyeNormPath):
            print("Segmentation process failed", file=sys.stderr)
            return None, None, None, None;

        try:
            eyeNorm = read_image(eyeNormPath)
            eye = read_image(eyePath)
            eyeNormMask = read_image(eyeNormMaskPath)
            eyeMask = read_image(eyeMaskPath)
        except:
            # delete temp. files
            shutil.rmtree(os.path.dirname(eyeNormPath))
        
        # delete temp. files
        shutil.rmtree(os.path.dirname(eyeNormPath))

        return eye, eyeNorm, eyeMask, eyeNormMask

if __name__ == '__main__':
    segmentator = Segmentator()
    segmentator.segment(sys.argv[1], **{"debug":True, "out":os.path.abspath("./.tmp"), "mode": "debug", "method": "hough"})