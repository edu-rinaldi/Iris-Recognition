# Demo

To test a subset of the proposed models (*"Hough approach"* for segmentation and *"FeatNet"* for feature extraction), I decided to develop a small test demo using scripts in Python. In the following directory you can find the following 3 executable scripts:
- "`Enrollment.py`": responsible for the enrollment of a subject
- "`Verification.py`": responsible for performing a verification operation
- "`Identification.py`": allows you to perform an identification operation.

## Before starting
Before using the various scripts, it is necessary to compile the segmenter executable. To do this, you can follow the instructions in "`SegmentationModule/README.md`".

Once you have obtained the executable, copy it into the folder "`Demo/Segmentation/bin/`".

## Enrollment
This script is used for enrolling a subject in the "database".
To run this process:
```bash
python Enrollment.py --in "path/to/irisImg" --id SUBJECT_ID --dataset "Storage/{datasetName}.csv"
```

Example:
```bash
python Enrollment.py --in "Temp/MyIris.png" --id 2 --dataset "Storage/MyDataset.csv"
```

## Verification
This script is used for verifying if the input image is an iris associated to the claimed identity.
To run this process:
```bash
python Verification.py --in "path/to/irisImg" --id CLAIMED_SUBJECT_ID --dataset "Storage/{datasetName}.csv"
```

Example:
```bash
python Verification.py --in "Temp/MyIris.png" --id 2 --dataset "Storage/MyDataset.csv"
```


## Identification

This script is used for identifying the identity of the subject associated to the input image.
To run this process:
```bash
python Identification.py --in "path/to/irisImg" --dataset "Storage/{datasetName}.csv" [--at ACCEPTANCE_THRESHOLD] 
```

Example:
```bash
python Identification.py --in "Temp/MyIris.png" --dataset "Storage/MyDataset.csv" --at 1.5
```
