/**
 * Copyright 2014 University of Bremen, Institute for Artificial Intelligence
 * Author: Thiemo Wiedemeyer <wiedemeyer@cs.uni-bremen.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <sys/stat.h>

#if defined(__linux__)
#include <sys/prctl.h>
#elif defined(__APPLE__)
#include <pthread.h>
#endif

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"

#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "tf2/LinearMath/Transform.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "geometry_msgs/msg/transform_stamped.hpp"

#include "kinect2_definitions.h"
#include "kinect2_registration/kinect2_registration.h"
#include "kinect2_registration/kinect2_console.h"

#include "libfreenect2/libfreenect2.hpp"
#include "libfreenect2/frame_listener_impl.h"
#include "libfreenect2/packet_pipeline.h"
#include "libfreenect2/config.h"
#include "libfreenect2/registration.h"

#include <chrono>

using namespace std::chrono_literals;

class Kinect2BridgeNode : public rclcpp::Node
{
private:
  std::vector<int> compressionParams;
  std::string compression16BitExt, compression16BitString, baseNameTF;

  cv::Size sizeColor, sizeIr, sizeLowRes;
  libfreenect2::Frame color;
  cv::Mat cameraMatrixColor, distortionColor, cameraMatrixLowRes, cameraMatrixIr, distortionIr, cameraMatrixDepth, distortionDepth;
  cv::Mat rotation, translation;
  cv::Mat map1Color, map2Color, map1Ir, map2Ir, map1LowRes, map2LowRes;

  std::vector<std::thread> threads;
  std::mutex lockIrDepth, lockColor;
  std::mutex lockSync, lockPub, lockTime, lockStatus;
  std::mutex lockRegLowRes, lockRegHighRes, lockRegSD;

  bool publishTF;
  std::thread tfPublisher, mainThread;

  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *device;
  libfreenect2::SyncMultiFrameListener *listenerColor, *listenerIrDepth;
  libfreenect2::PacketPipeline *packetPipeline;
  libfreenect2::Registration *registration;
  libfreenect2::Freenect2Device::ColorCameraParams colorParams;
  libfreenect2::Freenect2Device::IrCameraParams irParams;

  DepthRegistration *depthRegLowRes, *depthRegHighRes;

  size_t frameColor, frameIrDepth, pubFrameColor, pubFrameIrDepth;
  rclcpp::Time lastColor, lastDepth;

  bool nextColor, nextIrDepth;
  double deltaT, depthShift, elapsedTimeColor, elapsedTimeIrDepth;
  bool running, deviceActive, clientConnected, isSubscribedColor, isSubscribedDepth;

  enum Image
  {
    IR_SD = 0,
    IR_SD_RECT,

    DEPTH_SD,
    DEPTH_SD_RECT,
    DEPTH_HD,
    DEPTH_QHD,

    COLOR_SD_RECT,
    COLOR_HD,
    COLOR_HD_RECT,
    COLOR_QHD,
    COLOR_QHD_RECT,

    MONO_HD,
    MONO_HD_RECT,
    MONO_QHD,
    MONO_QHD_RECT,

    COUNT
  };

  enum Status
  {
    UNSUBCRIBED = 0,
    RAW,
    COMPRESSED,
    BOTH
  };

  std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> imagePubs;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr> compressedPubs;

  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr infoHDPub;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr infoQHDPub;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr infoIRPub;

  sensor_msgs::msg::CameraInfo infoHD,
      infoQHD, infoIR;
  std::vector<Status> status;

  rclcpp::TimerBase::SharedPtr timerCallback;

public:
  Kinect2BridgeNode()
      : rclcpp::Node("kinect2_bridge_node"), sizeColor(1920, 1080), sizeIr(512, 424), sizeLowRes(sizeColor.width / 2, sizeColor.height / 2), color(sizeColor.width, sizeColor.height, 4),
        frameColor(0), frameIrDepth(0), pubFrameColor(0), pubFrameIrDepth(0), lastColor(0, 0), lastDepth(0, 0), nextColor(false),
        nextIrDepth(false), depthShift(0), running(false), deviceActive(false), clientConnected(false)
  {
    status.resize(COUNT, UNSUBCRIBED);
  }

  bool start()
  {
    if (running)
    {
      OUT_ERROR(this, "kinect2_bridge is already running!");
      return false;
    }
    if (!initialize())
    {
      OUT_ERROR(this, "Initialization failed!");
      return false;
    }
    running = true;

    if (publishTF)
    {
      tfPublisher = std::thread(&Kinect2BridgeNode::publishStaticTF, this);
    }

    for (size_t i = 0; i < threads.size(); ++i)
    {
      threads[i] = std::thread(&Kinect2BridgeNode::threadDispatcher, this, i);
    }

    mainThread = std::thread(&Kinect2BridgeNode::main, this);
    return true;
  }

  void stop()
  {
    if (!running)
    {
      OUT_ERROR(this, "kinect2_bridge is not running!");
      return;
    }
    running = false;

    mainThread.join();

    for (size_t i = 0; i < threads.size(); ++i)
    {
      threads[i].join();
    }

    if (publishTF)
    {
      tfPublisher.join();
    }

    if (deviceActive && !device->stop())
    {
      OUT_ERROR(this, "could not stop device!");
    }

    if (!device->close())
    {
      OUT_ERROR(this, "could not close device!");
    }

    delete listenerIrDepth;
    delete listenerColor;
    delete registration;

    delete depthRegLowRes;
    delete depthRegHighRes;
  }

private:
  bool initialize()
  {
    double fps_limit, maxDepth, minDepth;
    bool use_png, bilateral_filter, edge_aware_filter;
    int32_t jpeg_quality, png_level, queueSize, reg_dev, depth_dev, worker_threads;
    std::string depth_method, reg_method, calib_path, sensor, base_name;

    std::string depthDefault = "cpu";
    std::string regDefault = "default";

#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
    depthDefault = "opengl";
#endif
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
    depthDefault = "opencl";
#endif
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
    depthDefault = "cuda";
#endif
#ifdef DEPTH_REG_OPENCL
    regDefault = "opencl";
#endif

    this->declare_parameter("base_name", std::string(K2_DEFAULT_NS));
    this->declare_parameter("sensor", std::string(""));
    this->declare_parameter("fps_limit", -1.0);
    this->declare_parameter("calib_path", std::string(K2_CALIB_PATH));
    this->declare_parameter("use_png", false);
    this->declare_parameter("jpeg_quality", 90);
    this->declare_parameter("png_level", 1);
    this->declare_parameter("depth_method", depthDefault);
    this->declare_parameter("depth_device", -1);
    this->declare_parameter("reg_method", regDefault);
    this->declare_parameter("reg_device", -1);
    this->declare_parameter("max_depth", 12.0);
    this->declare_parameter("min_depth", 0.1);
    this->declare_parameter("queue_size", 2);
    this->declare_parameter("bilateral_filter", true);
    this->declare_parameter("edge_aware_filter", true);
    this->declare_parameter("publish_tf", false);
    this->declare_parameter("base_name_tf", base_name);
    this->declare_parameter("worker_threads", 4);

    this->get_parameter("base_name", base_name);
    this->get_parameter("sensor", sensor);
    this->get_parameter("fps_limit", fps_limit);
    this->get_parameter("calib_path", calib_path);
    this->get_parameter("use_png", use_png);
    this->get_parameter("jpeg_quality", jpeg_quality);
    this->get_parameter("png_level", png_level);
    this->get_parameter("depth_method", depth_method);
    this->get_parameter("depth_device", depth_dev);
    this->get_parameter("reg_method", reg_method);
    this->get_parameter("reg_device", reg_dev);
    this->get_parameter("max_depth", maxDepth);
    this->get_parameter("min_depth", minDepth);
    this->get_parameter("queue_size", queueSize);
    this->get_parameter("bilateral_filter", bilateral_filter);
    this->get_parameter("edge_aware_filter", edge_aware_filter);
    this->get_parameter("publish_tf", publishTF);
    this->get_parameter("base_name_tf", baseNameTF);
    this->get_parameter("worker_threads", worker_threads);

    worker_threads = std::max(1, worker_threads);
    threads.resize(worker_threads);

    OUT_INFO(this, "parameter:" << std::endl
                                << "        base_name: " FG_CYAN << base_name << NO_COLOR << std::endl
                                << "           sensor: " FG_CYAN << (sensor.empty() ? "default" : sensor) << NO_COLOR << std::endl
                                << "        fps_limit: " FG_CYAN << fps_limit << NO_COLOR << std::endl
                                << "       calib_path: " FG_CYAN << calib_path << NO_COLOR << std::endl
                                << "          use_png: " FG_CYAN << (use_png ? "true" : "false") << NO_COLOR << std::endl
                                << "     jpeg_quality: " FG_CYAN << jpeg_quality << NO_COLOR << std::endl
                                << "        png_level: " FG_CYAN << png_level << NO_COLOR << std::endl
                                << "     depth_method: " FG_CYAN << depth_method << NO_COLOR << std::endl
                                << "     depth_device: " FG_CYAN << depth_dev << NO_COLOR << std::endl
                                << "       reg_method: " FG_CYAN << reg_method << NO_COLOR << std::endl
                                << "       reg_device: " FG_CYAN << reg_dev << NO_COLOR << std::endl
                                << "        max_depth: " FG_CYAN << maxDepth << NO_COLOR << std::endl
                                << "        min_depth: " FG_CYAN << minDepth << NO_COLOR << std::endl
                                << "       queue_size: " FG_CYAN << queueSize << NO_COLOR << std::endl
                                << " bilateral_filter: " FG_CYAN << (bilateral_filter ? "true" : "false") << NO_COLOR << std::endl
                                << "edge_aware_filter: " FG_CYAN << (edge_aware_filter ? "true" : "false") << NO_COLOR << std::endl
                                << "       publish_tf: " FG_CYAN << (publishTF ? "true" : "false") << NO_COLOR << std::endl
                                << "     base_name_tf: " FG_CYAN << baseNameTF << NO_COLOR << std::endl
                                << "   worker_threads: " FG_CYAN << worker_threads << NO_COLOR);

    deltaT = fps_limit > 0 ? 1.0 / fps_limit : 0.0;

    if (calib_path.empty() || calib_path.back() != '/')
    {
      calib_path += '/';
    }

    initCompression(jpeg_quality, png_level, use_png);

    if (!initPipeline(depth_method, depth_dev))
    {
      return false;
    }

    if (!initDevice(sensor))
    {
      return false;
    }

    initConfig(bilateral_filter, edge_aware_filter, minDepth, maxDepth);

    initCalibration(calib_path, sensor);

    if (!initRegistration(reg_method, reg_dev, maxDepth))
    {
      if (!device->close())
      {
        OUT_ERROR(this, "could not close device!");
      }
      delete listenerIrDepth;
      delete listenerColor;
      return false;
    }

    createCameraInfo();
    initTopics(queueSize, base_name);

    return true;
  }

  bool initRegistration(const std::string &method, const int32_t device, const double maxDepth)
  {
    DepthRegistration::Method reg;

    if (method == "default")
    {
      reg = DepthRegistration::DEFAULT;
    }
    else if (method == "cpu")
    {
#ifdef DEPTH_REG_CPU
      reg = DepthRegistration::CPU;
#else
      OUT_ERROR(this, "CPU registration is not available!");
      return false;
#endif
    }
    else if (method == "opencl")
    {
#ifdef DEPTH_REG_OPENCL
      reg = DepthRegistration::OPENCL;
#else
      OUT_ERROR(this, "OpenCL registration is not available!");
      return false;
#endif
    }
    else
    {
      OUT_ERROR(this, "Unknown registration method: " << method);
      return false;
    }

    depthRegLowRes = DepthRegistration::New(this->shared_from_this(), reg);
    depthRegHighRes = DepthRegistration::New(this->shared_from_this(), reg);

    if (!depthRegLowRes->init(cameraMatrixLowRes, sizeLowRes, cameraMatrixDepth, sizeIr, distortionDepth, rotation, translation, 0.5f, maxDepth, device) ||
        !depthRegHighRes->init(cameraMatrixColor, sizeColor, cameraMatrixDepth, sizeIr, distortionDepth, rotation, translation, 0.5f, maxDepth, device))
    {
      delete depthRegLowRes;
      delete depthRegHighRes;
      return false;
    }

    registration = new libfreenect2::Registration(irParams, colorParams);

    return true;
  }

  bool initPipeline(const std::string &method, const int32_t device)
  {
    if (method == "default")
    {
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
      packetPipeline = new libfreenect2::CudaPacketPipeline(device);
#elif defined(LIBFREENECT2_WITH_OPENCL_SUPPORT)
      packetPipeline = new libfreenect2::OpenCLPacketPipeline(device);
#elif defined(LIBFREENECT2_WITH_OPENGL_SUPPORT)
      packetPipeline = new libfreenect2::OpenGLPacketPipeline();
#else
      packetPipeline = new libfreenect2::CpuPacketPipeline();
#endif
    }
    else if (method == "cpu")
    {
      packetPipeline = new libfreenect2::CpuPacketPipeline();
    }
    else if (method == "cuda")
    {
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
      packetPipeline = new libfreenect2::CudaPacketPipeline(device);
#else
      OUT_ERROR(this, "Cuda depth processing is not available!");
      return false;
#endif
    }
    else if (method == "opencl")
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      packetPipeline = new libfreenect2::OpenCLPacketPipeline(device);
#else
      OUT_ERROR(this, "OpenCL depth processing is not available!");
      return false;
#endif
    }
    else if (method == "opengl")
    {
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
      packetPipeline = new libfreenect2::OpenGLPacketPipeline();
#else
      OUT_ERROR(this, "OpenGL depth processing is not available!");
      return false;
#endif
    }
    else if (method == "clkde")
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      packetPipeline = new libfreenect2::OpenCLKdePacketPipeline(device);
#else
      OUT_ERROR(this, "OpenCL depth processing is not available!");
      return false;
#endif
    }
    else if (method == "cudakde")
    {
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
      packetPipeline = new libfreenect2::CudaKdePacketPipeline(device);
#else
      OUT_ERROR(this, "Cuda depth processing is not available!");
      return false;
#endif
    }
    else
    {
      OUT_ERROR(this, "Unknown depth processing method: " << method);
      return false;
    }

    return true;
  }

  void initConfig(const bool bilateral_filter, const bool edge_aware_filter, const double minDepth, const double maxDepth)
  {
    libfreenect2::Freenect2Device::Config config;
    config.EnableBilateralFilter = bilateral_filter;
    config.EnableEdgeAwareFilter = edge_aware_filter;
    config.MinDepth = minDepth;
    config.MaxDepth = maxDepth;
    device->setConfiguration(config);
  }

  void initCompression(const int32_t jpegQuality, const int32_t pngLevel, const bool use_png)
  {
    compressionParams.resize(7, 0);
    compressionParams[0] = cv::IMWRITE_JPEG_QUALITY;
    compressionParams[1] = jpegQuality;
    compressionParams[2] = cv::IMWRITE_PNG_COMPRESSION;
    compressionParams[3] = pngLevel;
    compressionParams[4] = cv::IMWRITE_PNG_STRATEGY;
    compressionParams[5] = cv::IMWRITE_PNG_STRATEGY_RLE;
    compressionParams[6] = 0;

    if (use_png)
    {
      compression16BitExt = ".png";
      compression16BitString = std::string(sensor_msgs::image_encodings::TYPE_16UC1) + "; png compressed";
    }
    else
    {
      compression16BitExt = ".tif";
      compression16BitString = std::string(sensor_msgs::image_encodings::TYPE_16UC1) + "; tiff compressed";
    }
  }

  void initTopics(const int32_t queueSize, const std::string &base_name)
  {
    std::vector<std::string> topics(COUNT);
    topics[IR_SD] = K2_TOPIC_SD K2_TOPIC_IMAGE_IR;
    topics[IR_SD_RECT] = K2_TOPIC_SD K2_TOPIC_IMAGE_IR K2_TOPIC_IMAGE_RECT;

    topics[DEPTH_SD] = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH;
    topics[DEPTH_SD_RECT] = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
    topics[DEPTH_HD] = K2_TOPIC_HD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
    topics[DEPTH_QHD] = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;

    topics[COLOR_SD_RECT] = K2_TOPIC_SD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;

    topics[COLOR_HD] = K2_TOPIC_HD K2_TOPIC_IMAGE_COLOR;
    topics[COLOR_HD_RECT] = K2_TOPIC_HD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
    topics[COLOR_QHD] = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR;
    topics[COLOR_QHD_RECT] = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;

    topics[MONO_HD] = K2_TOPIC_HD K2_TOPIC_IMAGE_MONO;
    topics[MONO_HD_RECT] = K2_TOPIC_HD K2_TOPIC_IMAGE_MONO K2_TOPIC_IMAGE_RECT;
    topics[MONO_QHD] = K2_TOPIC_QHD K2_TOPIC_IMAGE_MONO;
    topics[MONO_QHD_RECT] = K2_TOPIC_QHD K2_TOPIC_IMAGE_MONO K2_TOPIC_IMAGE_RECT;

    imagePubs.resize(COUNT);
    compressedPubs.resize(COUNT);

    timerCallback = this->create_wall_timer(1000ms, [&]()
                                            { this->callbackStatus(); });

    for (size_t i = 0; i < COUNT; ++i)
    {
      imagePubs[i] = this->create_publisher<sensor_msgs::msg::Image>(base_name + topics[i], queueSize);
      compressedPubs[i] = this->create_publisher<sensor_msgs::msg::CompressedImage>(base_name + topics[i] + K2_TOPIC_COMPRESSED, queueSize);
    }

    infoHDPub = this->create_publisher<sensor_msgs::msg::CameraInfo>(base_name + K2_TOPIC_HD + K2_TOPIC_INFO, queueSize);
    infoQHDPub = this->create_publisher<sensor_msgs::msg::CameraInfo>(base_name + K2_TOPIC_QHD + K2_TOPIC_INFO, queueSize);
    infoIRPub = this->create_publisher<sensor_msgs::msg::CameraInfo>(base_name + K2_TOPIC_SD + K2_TOPIC_INFO, queueSize);
  }

  bool initDevice(std::string &sensor)
  {
    bool deviceFound = false;
    const int numOfDevs = freenect2.enumerateDevices();

    if (numOfDevs <= 0)
    {
      OUT_ERROR(this, "no Kinect2 devices found!");
      delete packetPipeline;
      return false;
    }

    if (sensor.empty())
    {
      sensor = freenect2.getDefaultDeviceSerialNumber();
    }

    OUT_INFO(this, "Kinect2 devices found: ");
    for (int i = 0; i < numOfDevs; ++i)
    {
      const std::string &s = freenect2.getDeviceSerialNumber(i);
      deviceFound = deviceFound || s == sensor;
      OUT_INFO(this, "  " << i << ": " FG_CYAN << s << (s == sensor ? FG_YELLOW " (selected)" : "") << NO_COLOR);
    }

    if (!deviceFound)
    {
      OUT_ERROR(this, "Device with serial '" << sensor << "' not found!");
      delete packetPipeline;
      return false;
    }

    device = freenect2.openDevice(sensor, packetPipeline);

    if (device == 0)
    {
      OUT_INFO(this, "no device connected or failure opening the default one!");
      return false;
    }

    listenerColor = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color);
    listenerIrDepth = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);

    device->setColorFrameListener(listenerColor);
    device->setIrAndDepthFrameListener(listenerIrDepth);

    OUT_INFO(this, "starting kinect2");
    if (!device->start())
    {
      OUT_ERROR(this, "could not start device!");
      delete listenerIrDepth;
      delete listenerColor;
      return false;
    }

    OUT_INFO(this, "device serial: " FG_CYAN << sensor << NO_COLOR);
    OUT_INFO(this, "device firmware: " FG_CYAN << device->getFirmwareVersion() << NO_COLOR);

    colorParams = device->getColorCameraParams();
    irParams = device->getIrCameraParams();

    if (!device->stop())
    {
      OUT_ERROR(this, "could not stop device!");
      delete listenerIrDepth;
      delete listenerColor;
      return false;
    }

    OUT_DEBUG(this, "default ir camera parameters: ");
    OUT_DEBUG(this, "fx: " FG_CYAN << irParams.fx << NO_COLOR ", fy: " FG_CYAN << irParams.fy << NO_COLOR ", cx: " FG_CYAN << irParams.cx << NO_COLOR ", cy: " FG_CYAN << irParams.cy << NO_COLOR);
    OUT_DEBUG(this, "k1: " FG_CYAN << irParams.k1 << NO_COLOR ", k2: " FG_CYAN << irParams.k2 << NO_COLOR ", p1: " FG_CYAN << irParams.p1 << NO_COLOR ", p2: " FG_CYAN << irParams.p2 << NO_COLOR ", k3: " FG_CYAN << irParams.k3 << NO_COLOR);

    OUT_DEBUG(this, "default color camera parameters: ");
    OUT_DEBUG(this, "fx: " FG_CYAN << colorParams.fx << NO_COLOR ", fy: " FG_CYAN << colorParams.fy << NO_COLOR ", cx: " FG_CYAN << colorParams.cx << NO_COLOR ", cy: " FG_CYAN << colorParams.cy << NO_COLOR);

    cameraMatrixColor = cv::Mat::eye(3, 3, CV_64F);
    distortionColor = cv::Mat::zeros(1, 5, CV_64F);

    cameraMatrixColor.at<double>(0, 0) = colorParams.fx;
    cameraMatrixColor.at<double>(1, 1) = colorParams.fy;
    cameraMatrixColor.at<double>(0, 2) = colorParams.cx;
    cameraMatrixColor.at<double>(1, 2) = colorParams.cy;
    cameraMatrixColor.at<double>(2, 2) = 1;

    cameraMatrixIr = cv::Mat::eye(3, 3, CV_64F);
    distortionIr = cv::Mat::zeros(1, 5, CV_64F);

    cameraMatrixIr.at<double>(0, 0) = irParams.fx;
    cameraMatrixIr.at<double>(1, 1) = irParams.fy;
    cameraMatrixIr.at<double>(0, 2) = irParams.cx;
    cameraMatrixIr.at<double>(1, 2) = irParams.cy;
    cameraMatrixIr.at<double>(2, 2) = 1;

    distortionIr.at<double>(0, 0) = irParams.k1;
    distortionIr.at<double>(0, 1) = irParams.k2;
    distortionIr.at<double>(0, 2) = irParams.p1;
    distortionIr.at<double>(0, 3) = irParams.p2;
    distortionIr.at<double>(0, 4) = irParams.k3;

    cameraMatrixDepth = cameraMatrixIr.clone();
    distortionDepth = distortionIr.clone();

    rotation = cv::Mat::eye(3, 3, CV_64F);
    translation = cv::Mat::zeros(3, 1, CV_64F);
    return true;
  }

  void initCalibration(const std::string &calib_path, const std::string &sensor)
  {
    std::string calibPath = calib_path + sensor + '/';

    struct stat fileStat;
    bool calibDirNotFound = stat(calibPath.c_str(), &fileStat) != 0 || !S_ISDIR(fileStat.st_mode);
    if (calibDirNotFound || !loadCalibrationFile(calibPath + K2_CALIB_COLOR, cameraMatrixColor, distortionColor))
    {
      OUT_WARN(this, "using sensor defaults for color intrinsic parameters.");
    }

    if (calibDirNotFound || !loadCalibrationFile(calibPath + K2_CALIB_IR, cameraMatrixDepth, distortionDepth))
    {
      OUT_WARN(this, "using sensor defaults for ir intrinsic parameters.");
    }

    if (calibDirNotFound || !loadCalibrationPoseFile(calibPath + K2_CALIB_POSE, rotation, translation))
    {
      OUT_WARN(this, "using defaults for rotation and translation.");
    }

    if (calibDirNotFound || !loadCalibrationDepthFile(calibPath + K2_CALIB_DEPTH, depthShift))
    {
      OUT_WARN(this, "using defaults for depth shift.");
      depthShift = 0.0;
    }

    cameraMatrixLowRes = cameraMatrixColor.clone();
    cameraMatrixLowRes.at<double>(0, 0) /= 2;
    cameraMatrixLowRes.at<double>(1, 1) /= 2;
    cameraMatrixLowRes.at<double>(0, 2) /= 2;
    cameraMatrixLowRes.at<double>(1, 2) /= 2;

    const int mapType = CV_16SC2;
    cv::initUndistortRectifyMap(cameraMatrixColor, distortionColor, cv::Mat(), cameraMatrixColor, sizeColor, mapType, map1Color, map2Color);
    cv::initUndistortRectifyMap(cameraMatrixIr, distortionIr, cv::Mat(), cameraMatrixIr, sizeIr, mapType, map1Ir, map2Ir);
    cv::initUndistortRectifyMap(cameraMatrixColor, distortionColor, cv::Mat(), cameraMatrixLowRes, sizeLowRes, mapType, map1LowRes, map2LowRes);

    OUT_DEBUG(this, "camera parameters used:");
    OUT_DEBUG(this, "camera matrix color:" FG_CYAN << std::endl
                                                   << cameraMatrixColor << NO_COLOR);
    OUT_DEBUG(this, "distortion coefficients color:" FG_CYAN << std::endl
                                                             << distortionColor << NO_COLOR);
    OUT_DEBUG(this, "camera matrix ir:" FG_CYAN << std::endl
                                                << cameraMatrixIr << NO_COLOR);
    OUT_DEBUG(this, "distortion coefficients ir:" FG_CYAN << std::endl
                                                          << distortionIr << NO_COLOR);
    OUT_DEBUG(this, "camera matrix depth:" FG_CYAN << std::endl
                                                   << cameraMatrixDepth << NO_COLOR);
    OUT_DEBUG(this, "distortion coefficients depth:" FG_CYAN << std::endl
                                                             << distortionDepth << NO_COLOR);
    OUT_DEBUG(this, "rotation:" FG_CYAN << std::endl
                                        << rotation << NO_COLOR);
    OUT_DEBUG(this, "translation:" FG_CYAN << std::endl
                                           << translation << NO_COLOR);
    OUT_DEBUG(this, "depth shift:" FG_CYAN << std::endl
                                           << depthShift << NO_COLOR);
  }

  bool loadCalibrationFile(const std::string &filename, cv::Mat &cameraMatrix, cv::Mat &distortion) const
  {
    cv::FileStorage fs;
    if (fs.open(filename, cv::FileStorage::READ))
    {
      fs[K2_CALIB_CAMERA_MATRIX] >> cameraMatrix;
      fs[K2_CALIB_DISTORTION] >> distortion;
      fs.release();
    }
    else
    {
      OUT_ERROR(this, "can't open calibration file: " << filename);
      return false;
    }
    return true;
  }

  bool loadCalibrationPoseFile(const std::string &filename, cv::Mat &rotation, cv::Mat &translation) const
  {
    cv::FileStorage fs;
    if (fs.open(filename, cv::FileStorage::READ))
    {
      fs[K2_CALIB_ROTATION] >> rotation;
      fs[K2_CALIB_TRANSLATION] >> translation;
      fs.release();
    }
    else
    {
      OUT_ERROR(this, "can't open calibration pose file: " << filename);
      return false;
    }
    return true;
  }

  bool loadCalibrationDepthFile(const std::string &filename, double &depthShift) const
  {
    cv::FileStorage fs;
    if (fs.open(filename, cv::FileStorage::READ))
    {
      fs[K2_CALIB_DEPTH_SHIFT] >> depthShift;
      fs.release();
    }
    else
    {
      OUT_ERROR(this, "can't open calibration depth file: " << filename);
      return false;
    }
    return true;
  }

  void createCameraInfo()
  {
    cv::Mat projColor = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat projIr = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat projLowRes = cv::Mat::zeros(3, 4, CV_64F);

    cameraMatrixColor.copyTo(projColor(cv::Rect(0, 0, 3, 3)));
    cameraMatrixIr.copyTo(projIr(cv::Rect(0, 0, 3, 3)));
    cameraMatrixLowRes.copyTo(projLowRes(cv::Rect(0, 0, 3, 3)));

    createCameraInfo(sizeColor, cameraMatrixColor, distortionColor, cv::Mat::eye(3, 3, CV_64F), projColor, infoHD);
    createCameraInfo(sizeIr, cameraMatrixIr, distortionIr, cv::Mat::eye(3, 3, CV_64F), projIr, infoIR);
    createCameraInfo(sizeLowRes, cameraMatrixLowRes, distortionColor, cv::Mat::eye(3, 3, CV_64F), projLowRes, infoQHD);
  }

  void createCameraInfo(const cv::Size &size, const cv::Mat &cameraMatrix, const cv::Mat &distortion, const cv::Mat &rotation, const cv::Mat &projection, sensor_msgs::msg::CameraInfo &cameraInfo) const
  {
    cameraInfo.height = size.height;
    cameraInfo.width = size.width;

    const double *itC = cameraMatrix.ptr<double>(0, 0);
    for (size_t i = 0; i < 9; ++i, ++itC)
    {
      cameraInfo.k[i] = *itC;
    }

    const double *itR = rotation.ptr<double>(0, 0);
    for (size_t i = 0; i < 9; ++i, ++itR)
    {
      cameraInfo.r[i] = *itR;
    }

    const double *itP = projection.ptr<double>(0, 0);
    for (size_t i = 0; i < 12; ++i, ++itP)
    {
      cameraInfo.p[i] = *itP;
    }

    cameraInfo.distortion_model = "plumb_bob";
    cameraInfo.d.resize(distortion.cols);
    const double *itD = distortion.ptr<double>(0, 0);
    for (size_t i = 0; i < (size_t)distortion.cols; ++i, ++itD)
    {
      cameraInfo.d[i] = *itD;
    }
  }

  void callbackStatus()
  {
    bool isSubscribedDepth = false;
    bool isSubscribedColor = false;

    lockStatus.lock();
    clientConnected = updateStatus(isSubscribedColor, isSubscribedDepth);
    bool error = false;

    if (clientConnected && !deviceActive)
    {
      OUT_INFO(this, "client connected. starting device...");
      if (!device->startStreams(isSubscribedColor, isSubscribedDepth))
      {
        OUT_ERROR(this, "could not start device!");
        error = true;
      }
      else
      {
        deviceActive = true;
      }
    }
    else if (deviceActive && (isSubscribedColor != this->isSubscribedColor || isSubscribedDepth != this->isSubscribedDepth))
    {
      if (!device->stop())
      {
        OUT_ERROR(this, "could not stop device!");
        error = true;
      }
      else if (!device->startStreams(isSubscribedColor, isSubscribedDepth))
      {
        OUT_ERROR(this, "could not start device!");
        error = true;
        deviceActive = false;
      }
    }
    this->isSubscribedColor = isSubscribedColor;
    this->isSubscribedDepth = isSubscribedDepth;
    lockStatus.unlock();

    if (error)
    {
      stop();
    }
  }

  bool updateStatus(bool &isSubscribedColor, bool &isSubscribedDepth)
  {
    isSubscribedDepth = false;
    isSubscribedColor = false;

    for (size_t i = 0; i < COUNT; ++i)
    {
      Status s = UNSUBCRIBED;
      if (imagePubs[i]->get_subscription_count() > 0)
      {
        s = RAW;
      }
      if (compressedPubs[i]->get_subscription_count() > 0)
      {
        s = s == RAW ? BOTH : COMPRESSED;
      }

      if (i <= COLOR_SD_RECT && s != UNSUBCRIBED)
      {
        isSubscribedDepth = true;
      }
      if (i >= COLOR_SD_RECT && s != UNSUBCRIBED)
      {
        isSubscribedColor = true;
      }

      status[i] = s;
    }
    if (infoHDPub->get_subscription_count() > 0 || infoQHDPub->get_subscription_count() > 0)
    {
      isSubscribedColor = true;
    }
    if (infoIRPub->get_subscription_count() > 0)
    {
      isSubscribedDepth = true;
    }

    return isSubscribedColor || isSubscribedDepth;
  }

  void main()
  {
    setThreadName("Controll");
    OUT_INFO(this, "waiting for clients to connect");
    double nextFrame = this->get_clock()->now().seconds() + deltaT;
    double fpsTime = this->get_clock()->now().seconds();
    size_t oldFrameIrDepth = 0, oldFrameColor = 0;
    nextColor = true;
    nextIrDepth = true;

    while (running)
    {
      if (!deviceActive)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        fpsTime = this->get_clock()->now().seconds();
        nextFrame = fpsTime + deltaT;
        continue;
      }

      double now = this->get_clock()->now().seconds();

      if (now - fpsTime >= 3.0)
      {
        fpsTime = now - fpsTime;
        size_t framesIrDepth = frameIrDepth - oldFrameIrDepth;
        size_t framesColor = frameColor - oldFrameColor;
        oldFrameIrDepth = frameIrDepth;
        oldFrameColor = frameColor;

        lockTime.lock();
        double tColor = elapsedTimeColor;
        double tDepth = elapsedTimeIrDepth;
        elapsedTimeColor = 0;
        elapsedTimeIrDepth = 0;
        lockTime.unlock();

        if (isSubscribedDepth)
        {
          OUT_INFO(this, "depth processing: " FG_YELLOW "~" << (tDepth / framesIrDepth) * 1000 << "ms" NO_COLOR " (~" << framesIrDepth / tDepth << "Hz) publishing rate: " FG_YELLOW "~" << framesIrDepth / fpsTime << "Hz" NO_COLOR);
        }
        if (isSubscribedColor)
        {
          OUT_INFO(this, "color processing: " FG_YELLOW "~" << (tColor / framesColor) * 1000 << "ms" NO_COLOR " (~" << framesColor / tColor << "Hz) publishing rate: " FG_YELLOW "~" << framesColor / fpsTime << "Hz" NO_COLOR);
        }
        fpsTime = now;
      }

      if (now >= nextFrame)
      {
        nextColor = true;
        nextIrDepth = true;
        nextFrame += deltaT;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));

      if (!deviceActive)
      {
        oldFrameIrDepth = frameIrDepth;
        oldFrameColor = frameColor;
        lockTime.lock();
        elapsedTimeColor = 0;
        elapsedTimeIrDepth = 0;
        lockTime.unlock();
        continue;
      }
    }
  }

  void threadDispatcher(const size_t id)
  {
    setThreadName("Worker" + std::to_string(id));
    const size_t checkFirst = id % 2;
    bool processedFrame = false;
    int oldNice = nice(0);
    oldNice = nice(19 - oldNice);

    while (running)
    {
      processedFrame = false;

      for (size_t i = 0; i < 2; ++i)
      {
        if (i == checkFirst)
        {
          if (nextIrDepth && lockIrDepth.try_lock())
          {
            nextIrDepth = false;
            receiveIrDepth();
            processedFrame = true;
          }
        }
        else
        {
          if (nextColor && lockColor.try_lock())
          {
            nextColor = false;
            receiveColor();
            processedFrame = true;
          }
        }
      }

      if (!processedFrame)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  void receiveIrDepth()
  {
    libfreenect2::FrameMap frames;
    cv::Mat depth, ir;
    std_msgs::msg::Header header;
    std::vector<cv::Mat> images(COUNT);
    std::vector<Status> status = this->status;
    size_t frame;

    if (!receiveFrames(listenerIrDepth, frames))
    {
      lockIrDepth.unlock();
      return;
    }
    double now = this->get_clock()->now().seconds();

    header = createHeader(lastDepth, lastColor);

    libfreenect2::Frame *irFrame = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depthFrame = frames[libfreenect2::Frame::Depth];

    if (irFrame->status != 0 || depthFrame->status != 0)
    {
      listenerIrDepth->release(frames);
      lockIrDepth.unlock();
      running = false;
      OUT_ERROR(this, "failure in depth packet processor from libfreenect2");
      return;
    }
    if (irFrame->format != libfreenect2::Frame::Float || depthFrame->format != libfreenect2::Frame::Float)
    {
      listenerIrDepth->release(frames);
      lockIrDepth.unlock();
      running = false;
      OUT_ERROR(this, "received invalid frame format");
      return;
    }

    frame = frameIrDepth++;

    if (status[COLOR_SD_RECT] || status[DEPTH_SD] || status[DEPTH_SD_RECT] || status[DEPTH_QHD] || status[DEPTH_HD])
    {
      cv::Mat(depthFrame->height, depthFrame->width, CV_32FC1, depthFrame->data).copyTo(depth);
    }

    if (status[IR_SD] || status[IR_SD_RECT])
    {
      ir = cv::Mat(irFrame->height, irFrame->width, CV_32FC1, irFrame->data);
      ir.convertTo(images[IR_SD], CV_16U);
    }

    listenerIrDepth->release(frames);
    lockIrDepth.unlock();

    processIrDepth(depth, images, status);

    publishImages(images, header, status, frame, pubFrameIrDepth, IR_SD, COLOR_HD);

    double elapsed = this->get_clock()->now().seconds() - now;
    lockTime.lock();
    elapsedTimeIrDepth += elapsed;
    lockTime.unlock();
  }

  void receiveColor()
  {
    libfreenect2::FrameMap frames;
    std_msgs::msg::Header header;
    std::vector<cv::Mat> images(COUNT);
    std::vector<Status> status = this->status;
    size_t frame;

    if (!receiveFrames(listenerColor, frames))
    {
      lockColor.unlock();
      return;
    }
    double now = this->get_clock()->now().seconds();

    header = createHeader(lastColor, lastDepth);

    libfreenect2::Frame *colorFrame = frames[libfreenect2::Frame::Color];

    if (colorFrame->status != 0)
    {
      listenerColor->release(frames);
      lockIrDepth.unlock();
      running = false;
      OUT_ERROR(this, "failure in rgb packet processor from libfreenect2");
      return;
    }
    if (colorFrame->format != libfreenect2::Frame::BGRX && colorFrame->format != libfreenect2::Frame::RGBX)
    {
      listenerColor->release(frames);
      lockIrDepth.unlock();
      running = false;
      OUT_ERROR(this, "received invalid frame format");
      return;
    }

    frame = frameColor++;

    cv::Mat color = cv::Mat(colorFrame->height, colorFrame->width, CV_8UC4, colorFrame->data);
    if (status[COLOR_SD_RECT])
    {
      lockRegSD.lock();
      memcpy(this->color.data, colorFrame->data, sizeColor.width * sizeColor.height * 4);
      this->color.format = colorFrame->format;
      lockRegSD.unlock();
    }
    if (status[COLOR_HD] || status[COLOR_HD_RECT] || status[COLOR_QHD] || status[COLOR_QHD_RECT] ||
        status[MONO_HD] || status[MONO_HD_RECT] || status[MONO_QHD] || status[MONO_QHD_RECT])
    {
      cv::Mat tmp;
      cv::flip(color, tmp, 1);
      if (colorFrame->format == libfreenect2::Frame::BGRX)
      {
        cv::cvtColor(tmp, images[COLOR_HD], cv::COLOR_BGRA2BGR);
      }
      else
      {
        cv::cvtColor(tmp, images[COLOR_HD], cv::COLOR_RGBA2BGR);
      }
    }

    listenerColor->release(frames);
    lockColor.unlock();

    processColor(images, status);

    publishImages(images, header, status, frame, pubFrameColor, COLOR_HD, COUNT);

    double elapsed = this->get_clock()->now().seconds() - now;
    lockTime.lock();
    elapsedTimeColor += elapsed;
    lockTime.unlock();
  }

  bool receiveFrames(libfreenect2::SyncMultiFrameListener *listener, libfreenect2::FrameMap &frames)
  {
    bool newFrames = false;
    for (; !newFrames;)
    {
#ifdef LIBFREENECT2_THREADING_STDLIB
      newFrames = listener->waitForNewFrame(frames, 1000);
#else
      newFrames = true;
      listener->waitForNewFrame(frames);
#endif
      if (!deviceActive || !running)
      {
        if (newFrames)
        {
          listener->release(frames);
        }
        return false;
      }
    }
    return newFrames;
  }

  std_msgs::msg::Header createHeader(rclcpp::Time &last, rclcpp::Time &other)
  {
    auto timestamp = this->get_clock()->now();
    lockSync.lock();
    if (other.seconds() == 0.0)
    {
      last = timestamp;
    }
    else
    {
      timestamp = other;
      other = rclcpp::Time(0, 0);
    }
    lockSync.unlock();

    std_msgs::msg::Header header;
    header.stamp = timestamp;
    return header;
  }

  void processIrDepth(const cv::Mat &depth, std::vector<cv::Mat> &images, const std::vector<Status> &status)
  {
    // COLOR registered to depth
    if (status[COLOR_SD_RECT])
    {
      cv::Mat tmp;
      libfreenect2::Frame depthFrame(sizeIr.width, sizeIr.height, 4, depth.data);
      libfreenect2::Frame undistorted(sizeIr.width, sizeIr.height, 4);
      libfreenect2::Frame registered(sizeIr.width, sizeIr.height, 4);
      lockRegSD.lock();
      registration->apply(&color, &depthFrame, &undistorted, &registered);
      lockRegSD.unlock();
      cv::flip(cv::Mat(sizeIr, CV_8UC4, registered.data), tmp, 1);
      if (color.format == libfreenect2::Frame::BGRX)
      {
        cv::cvtColor(tmp, images[COLOR_SD_RECT], cv::COLOR_BGRA2BGR);
      }
      else
      {
        cv::cvtColor(tmp, images[COLOR_SD_RECT], cv::COLOR_RGBA2BGR);
      }
    }

    // IR
    if (status[IR_SD] || status[IR_SD_RECT])
    {
      cv::flip(images[IR_SD], images[IR_SD], 1);
    }
    if (status[IR_SD_RECT])
    {
      cv::remap(images[IR_SD], images[IR_SD_RECT], map1Ir, map2Ir, cv::INTER_AREA);
    }

    // DEPTH
    cv::Mat depthShifted;
    if (status[DEPTH_SD])
    {
      depth.convertTo(images[DEPTH_SD], CV_16U, 1);
      cv::flip(images[DEPTH_SD], images[DEPTH_SD], 1);
    }
    if (status[DEPTH_SD_RECT] || status[DEPTH_QHD] || status[DEPTH_HD])
    {
      depth.convertTo(depthShifted, CV_16U, 1, depthShift);
      cv::flip(depthShifted, depthShifted, 1);
    }
    if (status[DEPTH_SD_RECT])
    {
      cv::remap(depthShifted, images[DEPTH_SD_RECT], map1Ir, map2Ir, cv::INTER_NEAREST);
    }
    if (status[DEPTH_QHD])
    {
      lockRegLowRes.lock();
      depthRegLowRes->registerDepth(depthShifted, images[DEPTH_QHD]);
      lockRegLowRes.unlock();
    }
    if (status[DEPTH_HD])
    {
      lockRegHighRes.lock();
      depthRegHighRes->registerDepth(depthShifted, images[DEPTH_HD]);
      lockRegHighRes.unlock();
    }
  }

  void processColor(std::vector<cv::Mat> &images, const std::vector<Status> &status)
  {
    // COLOR
    if (status[COLOR_HD_RECT] || status[MONO_HD_RECT])
    {
      cv::remap(images[COLOR_HD], images[COLOR_HD_RECT], map1Color, map2Color, cv::INTER_AREA);
    }
    if (status[COLOR_QHD] || status[MONO_QHD])
    {
      cv::resize(images[COLOR_HD], images[COLOR_QHD], sizeLowRes, 0, 0, cv::INTER_AREA);
    }
    if (status[COLOR_QHD_RECT] || status[MONO_QHD_RECT])
    {
      cv::remap(images[COLOR_HD], images[COLOR_QHD_RECT], map1LowRes, map2LowRes, cv::INTER_AREA);
    }

    // MONO
    if (status[MONO_HD])
    {
      cv::cvtColor(images[COLOR_HD], images[MONO_HD], cv::COLOR_BGR2GRAY);
    }
    if (status[MONO_HD_RECT])
    {
      cv::cvtColor(images[COLOR_HD_RECT], images[MONO_HD_RECT], cv::COLOR_BGR2GRAY);
    }
    if (status[MONO_QHD])
    {
      cv::cvtColor(images[COLOR_QHD], images[MONO_QHD], cv::COLOR_BGR2GRAY);
    }
    if (status[MONO_QHD_RECT])
    {
      cv::cvtColor(images[COLOR_QHD_RECT], images[MONO_QHD_RECT], cv::COLOR_BGR2GRAY);
    }
  }

  void publishImages(const std::vector<cv::Mat> &images, const std_msgs::msg::Header &header, const std::vector<Status> &status, const size_t frame, size_t &pubFrame, const size_t begin, const size_t end)
  {
    std::vector<sensor_msgs::msg::Image::SharedPtr> imageMsgs(COUNT);
    std::vector<sensor_msgs::msg::CompressedImage::SharedPtr> compressedMsgs(COUNT);
    sensor_msgs::msg::CameraInfo::SharedPtr infoHDMsg, infoQHDMsg, infoIRMsg;
    std_msgs::msg::Header _header = header;

    if (begin < COLOR_HD)
    {
      _header.frame_id = baseNameTF + K2_TF_IR_OPT_FRAME;

      infoIRMsg = std::make_shared<sensor_msgs::msg::CameraInfo>();
      *infoIRMsg = infoIR;
      infoIRMsg->header = _header;
    }
    else
    {
      _header.frame_id = baseNameTF + K2_TF_RGB_OPT_FRAME;

      infoHDMsg = std::make_shared<sensor_msgs::msg::CameraInfo>();
      *infoHDMsg = infoHD;
      infoHDMsg->header = _header;

      infoQHDMsg = std::make_shared<sensor_msgs::msg::CameraInfo>();
      *infoQHDMsg = infoQHD;
      infoQHDMsg->header = _header;
    }

    for (size_t i = begin; i < end; ++i)
    {
      if (i < DEPTH_HD || i == COLOR_SD_RECT)
      {
        _header.frame_id = baseNameTF + K2_TF_IR_OPT_FRAME;
      }
      else
      {
        _header.frame_id = baseNameTF + K2_TF_RGB_OPT_FRAME;
      }

      switch (status[i])
      {
      case UNSUBCRIBED:
        break;
      case RAW:
        imageMsgs[i] = std::make_shared<sensor_msgs::msg::Image>();
        createImage(images[i], _header, Image(i), imageMsgs[i]);
        break;
      case COMPRESSED:
        compressedMsgs[i] = std::make_shared<sensor_msgs::msg::CompressedImage>();
        createCompressed(images[i], _header, Image(i), compressedMsgs[i]);
        break;
      case BOTH:
        imageMsgs[i] = std::make_shared<sensor_msgs::msg::Image>();
        compressedMsgs[i] = std::make_shared<sensor_msgs::msg::CompressedImage>();
        createImage(images[i], _header, Image(i), imageMsgs[i]);
        createCompressed(images[i], _header, Image(i), compressedMsgs[i]);
        break;
      }
    }

    while (frame != pubFrame)
    {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    lockPub.lock();
    for (size_t i = begin; i < end; ++i)
    {
      switch (status[i])
      {
      case UNSUBCRIBED:
        break;
      case RAW:
        imagePubs[i]->publish(*imageMsgs[i]);
        break;
      case COMPRESSED:
        compressedPubs[i]->publish(*compressedMsgs[i]);
        break;
      case BOTH:
        imagePubs[i]->publish(*imageMsgs[i]);
        compressedPubs[i]->publish(*compressedMsgs[i]);
        break;
      }
    }

    if (begin < COLOR_HD)
    {
      if (infoIRPub->get_subscription_count() > 0)
      {
        infoIRPub->publish(*infoIRMsg);
      }
    }
    else
    {
      if (infoHDPub->get_subscription_count() > 0)
      {
        infoHDPub->publish(*infoHDMsg);
      }
      if (infoQHDPub->get_subscription_count() > 0)
      {
        infoQHDPub->publish(*infoQHDMsg);
      }
    }

    ++pubFrame;
    lockPub.unlock();
  }

  void createImage(const cv::Mat &image, const std_msgs::msg::Header &header, const Image type, sensor_msgs::msg::Image::SharedPtr &msgImage) const
  {
    size_t step, size;
    step = image.cols * image.elemSize();
    size = image.rows * step;

    switch (type)
    {
    case IR_SD:
    case IR_SD_RECT:
    case DEPTH_SD:
    case DEPTH_SD_RECT:
    case DEPTH_HD:
    case DEPTH_QHD:
      msgImage->encoding = sensor_msgs::image_encodings::TYPE_16UC1;
      break;
    case COLOR_SD_RECT:
    case COLOR_HD:
    case COLOR_HD_RECT:
    case COLOR_QHD:
    case COLOR_QHD_RECT:
      msgImage->encoding = sensor_msgs::image_encodings::BGR8;
      break;
    case MONO_HD:
    case MONO_HD_RECT:
    case MONO_QHD:
    case MONO_QHD_RECT:
      msgImage->encoding = sensor_msgs::image_encodings::TYPE_8UC1;
      break;
    case COUNT:
      return;
    }

    msgImage->header = header;
    msgImage->height = image.rows;
    msgImage->width = image.cols;
    msgImage->is_bigendian = false;
    msgImage->step = step;
    msgImage->data.resize(size);
    memcpy(msgImage->data.data(), image.data, size);
  }

  void createCompressed(const cv::Mat &image, const std_msgs::msg::Header &header, const Image type, sensor_msgs::msg::CompressedImage::SharedPtr &msgImage) const
  {
    msgImage->header = header;

    switch (type)
    {
    case IR_SD:
    case IR_SD_RECT:
    case DEPTH_SD:
    case DEPTH_SD_RECT:
    case DEPTH_HD:
    case DEPTH_QHD:
      msgImage->format = compression16BitString;
      cv::imencode(compression16BitExt, image, msgImage->data, compressionParams);
      break;
    case COLOR_SD_RECT:
    case COLOR_HD:
    case COLOR_HD_RECT:
    case COLOR_QHD:
    case COLOR_QHD_RECT:
      msgImage->format = std::string(sensor_msgs::image_encodings::BGR8) + "; jpeg compressed bgr8";
      cv::imencode(".jpg", image, msgImage->data, compressionParams);
      break;
    case MONO_HD:
    case MONO_HD_RECT:
    case MONO_QHD:
    case MONO_QHD_RECT:
      msgImage->format = std::string(sensor_msgs::image_encodings::TYPE_8UC1) + "; jpeg compressed ";
      cv::imencode(".jpg", image, msgImage->data, compressionParams);
      break;
    case COUNT:
      return;
    }
  }

  void publishStaticTF()
  {
    setThreadName("TFPublisher");
    tf2_ros::StaticTransformBroadcaster broadcaster(this);
    geometry_msgs::msg::TransformStamped stColorOpt, stIrOpt;
    auto now = this->get_clock()->now();

    tf2::Matrix3x3 rot(rotation.at<double>(0, 0), rotation.at<double>(0, 1), rotation.at<double>(0, 2),
                       rotation.at<double>(1, 0), rotation.at<double>(1, 1), rotation.at<double>(1, 2),
                       rotation.at<double>(2, 0), rotation.at<double>(2, 1), rotation.at<double>(2, 2));

    tf2::Quaternion qZero;
    qZero.setRPY(0, 0, 0);
    tf2::Vector3 trans(translation.at<double>(0), translation.at<double>(1), translation.at<double>(2));
    tf2::Vector3 vZero(0, 0, 0);
    tf2::Transform tIr(rot, trans), tZero(qZero, vZero);

    stColorOpt.transform = tf2::toMsg(tZero);
    stColorOpt.header.stamp = now;
    stColorOpt.child_frame_id = baseNameTF + K2_TF_RGB_OPT_FRAME;
    stColorOpt.header.frame_id = baseNameTF + K2_TF_LINK;

    stIrOpt.transform = tf2::toMsg(tIr);
    stIrOpt.header.stamp = now;
    stIrOpt.child_frame_id = baseNameTF + K2_TF_IR_OPT_FRAME;
    stIrOpt.header.frame_id = baseNameTF + K2_TF_RGB_OPT_FRAME;

    while (running)
    {
      now = this->get_clock()->now();
      stColorOpt.header.stamp = now;
      stIrOpt.header.stamp = now;

      broadcaster.sendTransform(stColorOpt);
      broadcaster.sendTransform(stIrOpt);

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  static inline void setThreadName(const std::string &name)
  {
#if defined(__linux__)
    prctl(PR_SET_NAME, name.c_str());
#elif defined(__APPLE__)
    pthread_setname_np(name.c_str());
#endif
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto bridge = std::make_shared<Kinect2BridgeNode>();
  bridge->start();

  rclcpp::spin(bridge);

  bridge->stop();

  rclcpp::shutdown();
  return 0;
}
