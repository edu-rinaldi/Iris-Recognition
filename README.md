# Iris-Recognition
This is a project which implements ISis v2 segmentation module and combined operators LBP and Spatiogram for recognition.

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
