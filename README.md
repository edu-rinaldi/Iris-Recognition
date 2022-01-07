# Iris-Recognition
Iris recognition system based on:
* ISis v2 segmentation method
* Hough gradient segmentation method
* LBP Operator for feature extraction

## How to install
I reccomend using CMake-GUI because it's easier, anyway you must have [OpenCV](https://opencv.org/) installed on your machine.
Then just:
```bash
git clone https://github.com/edu-rinaldi/Iris-Recognition && cd Iris-Recognition
mkdir build && cd build
cmake ..
cmake --build .
```

Or if you want to use Xcode, VS or other generator:

```bash
git clone https://github.com/edu-rinaldi/Iris-Recognition && cd Iris-Recognition
mkdir build && cd build
cmake -G [generator_name] ..
```
