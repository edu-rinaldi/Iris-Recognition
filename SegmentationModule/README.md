# Segmentation module
Unlike the others, I decided to create this module in C++17 because it allows me to carry out numerous low-level optimisations.

Two segmentation modules have been implemented:
1. *"ISis v.2"*, code available in the directory "`SegmentationModule/Segmentator/src/Isis`".
2. *"Hough approach"*, code available in the directory "`SegmentationModule/Segmentator/src/Hough`".

A high level description of how they work is available in the directory "`Report`", basically both implement an interface described in "`SegmentationModule/Segmentator/src/Segmentation.h`".


## Build
The only external dependency is the **OpenCV** library, which must be installed on your computer, so first thing download and install OpenCV.

You can build the project in two ways:

Using Cmake on terminal/cmd:
```bash
# create a build folder
mkdir build && cd build
cmake ..
cmake --build .
```

Or using CMake-GUI:
1. Select source code directory (this where I put this README file)
2. Create a "build" directory and select it in "Where to build binaries"
3. "Configure" --> select a generator, for windows I suggest MSVC
4. "Generate"

Now you can open the project/solution and compile.

If any of this step fails it's probably due to the fact that CMake didn't found OpenCV, so you must set "`OpenCV_DIR`" variable.

For building this project I suggest to use [*CMake-GUI*](https://cmake.org/download/) because it makes this process easier and more intuitive.