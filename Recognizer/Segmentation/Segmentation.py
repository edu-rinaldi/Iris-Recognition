import sys
import os
import subprocess

class Segmentator:
    def __init__(self):
        self.path = "./bin/SegmentatorApp.exe"

    def segment(self, imagePath, **kwargs):
        command = f'{self.path} --in "{imagePath}" '
        flagMode = False
        if 'method' in kwargs:
            command += f'--method "{kwargs["method"]}" '
        if 'size' in kwargs:
            command += f'--size "{kwargs["size"]}" '
        if 'mode' in kwargs:
            command += f'--mode "{kwargs["mode"]}" '
            flagMode = kwargs["mode"] == "segmentation"
        if 'out' in kwargs and flagMode:
            command += f'--out "{kwargs["out"]}" '
        if 'debug' in kwargs and kwargs["debug"]:
            print(f"Launching command: {command}")
        origWD = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.Popen(command)
        proc.wait()
        os.chdir(origWD)