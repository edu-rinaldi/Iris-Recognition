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

## How to use it
At the moment there're some bug with the paths, so to be sure that everything work, put your terminal in the executable folder (ex: Release, Debug, ...) before running the code.

### Enrolling
If you want to enroll an iris into the system, you can just run the code with `-e` flag:
```bash
./iris -e path/to/img
```

For example:
```bash
./iris -e /usr/foo/Desktop/my_iris.png
```

### Recognize
If you want to get the top 5 most similar iris to the one given in input you can run the code with `-r` flag:
```bash
./iris -r path/to/img
```

For example:
```bash
./iris -r /usr/foo/Desktop/my_iris.png
```

### Visualize segmentation
If you want to visualize the segmentation that you can obtain from the image, you can run the the code with `-ds` flag:
```bash
./iris -ds path/to/img
```

For example:
```bash
./iris -ds /usr/foo/Desktop/my_iris.png
```

### Print usage
You can print the usage with `-h` flag:
```bash
./iris -h
```