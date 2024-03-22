# Kinect2 ROS2

This package enables the usage of the Microsoft Kinect2 ('Kinect One') with ROS2 Humble.

This is a fork of of the [iai_kinect2 for OpenCV 4](https://github.com/paul-shuvo/iai_kinect2_opencv4/tree/master) package, a fork of the original [iai_kinect2](https://github.com/code-iai/iai_kinect2).

### What works?
The ```kinect2_bridge```, ```kinect2_registration``` and ```kinect2_calibration``` should be functional, although with a reduced feature set (e.g., no CUDA/OpenCL support).
The ```kinect2_viewer``` has been dropped completely.

### Prerequisites
Install [libfreenect](https://github.com/OpenKinect/libfreenect2)
```
git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2 && mkdir build && cd build
cmake .. -DENABLE_CXX11=ON -DBUILD_OPENNI2_DRIVER=OFF -DENABLE_OPENCL=OFF -DENABLE_CUDA=OFF -DENABLE_OPENGL=OFF -DENABLE_VAAPI=OFF -DENABLE_TEGRAJPEG=OFF -DCMAKE_INSTALL_PREFIX=/usr
sudo make install
```

### Install
Clone this report into your ```src``` and build.

```
git clone https://github.com/krepa098/kinect2_ros2.git
```

### Run
Launch the ```kinect2_bridge``` to receive ```color```, ```depth```, and ```mono``` images as well as the pointcloud.

```
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml
```

You can also launch ```rtabmap``` to create 3D scans:

```
ros2 launch kinect2_bridge rtabmap.launch.py
```

### Calibrate
Use the calibration utility
```
ros2 run kinect2_calibration kinect2_calibration_node <args>
```

See also [kinect2_calibration](./kinect2_calibration/).
