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
#include <fstream>
#include <string>
#include <vector>
#include <mutex>
#include <thread>

#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "cv_bridge/cv_bridge.h"

#include "image_transport/image_transport.hpp"
#include "image_transport/subscriber_filter.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "kinect2_calibration/kinect2_calibration_definitions.h"
#include "kinect2_bridge/kinect2_definitions.h"

using ExecutorSharedPtr = rclcpp::executors::MultiThreadedExecutor::SharedPtr;

enum Mode
{
  RECORD,
  CALIBRATE
};

enum Source
{
  COLOR,
  IR,
  SYNC
};

class Recorder
{
private:
  rclcpp::Node::SharedPtr node;
  ExecutorSharedPtr executor;

  const bool circleBoard;
  int circleFlags;

  const cv::Size boardDims;
  const float boardSize;
  const Source mode;

  const std::string path;
  const std::string topicColor, topicIr, topicDepth;
  std::mutex lock;

  bool update;
  bool foundColor, foundIr;
  cv::Mat color, ir, irGrey, depth;

  size_t frame;
  std::vector<int> params;

  std::vector<cv::Point3f> board;
  std::vector<cv::Point2f> pointsColor, pointsIr;

  typedef message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image> ColorIrDepthSyncPolicy;

  image_transport::ImageTransport it;
  image_transport::SubscriberFilter *subImageColor, *subImageIr, *subImageDepth;
  message_filters::Synchronizer<ColorIrDepthSyncPolicy> *sync;

  int minIr, maxIr;
  cv::Ptr<cv::CLAHE> clahe;

public:
  Recorder(const rclcpp::Node::SharedPtr &node, const ExecutorSharedPtr &executor, const std::string &path, const std::string &topicColor, const std::string &topicIr, const std::string &topicDepth,
           const Source mode, const bool circleBoard, const bool symmetric, const cv::Size &boardDims, const float boardSize)
      : node(node), executor(executor), circleBoard(circleBoard), boardDims(boardDims), boardSize(boardSize), mode(mode), path(path), topicColor(topicColor), topicIr(topicIr),
        topicDepth(topicDepth), update(false), foundColor(false), foundIr(false), frame(0), it(node), minIr(0), maxIr(0x7FFF)
  {
    if (symmetric)
    {
      circleFlags = cv::CALIB_CB_SYMMETRIC_GRID + cv::CALIB_CB_CLUSTERING;
    }
    else
    {
      circleFlags = cv::CALIB_CB_ASYMMETRIC_GRID + cv::CALIB_CB_CLUSTERING;
    }

    params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    params.push_back(9);

    board.resize(boardDims.width * boardDims.height);
    if (symmetric)
    {
      for (size_t r = 0, i = 0; r < (size_t)boardDims.height; ++r)
      {
        for (size_t c = 0; c < (size_t)boardDims.width; ++c, ++i)
        {
          board[i] = cv::Point3f(c * boardSize, r * boardSize, 0);
        }
      }
    }
    else
    {
      for (size_t r = 0, i = 0; r < (size_t)boardDims.height; ++r)
      {
        for (size_t c = 0; c < (size_t)boardDims.width; ++c, ++i)
        {
          board[i] = cv::Point3f(float((2 * c + r % 2) * boardSize), float(r * boardSize), 0); // for asymmetrical circles
        }
      }
    }

    clahe = cv::createCLAHE(1.5, cv::Size(32, 32));
  }
  ~Recorder()
  {
  }

  void run()
  {
    startRecord();

    display();

    stopRecord();
  }

private:
  void startRecord()
  {
    OUT_INFO(node, "Controls:" << std::endl
                               << FG_YELLOW "   [ESC, q]" NO_COLOR " - Exit" << std::endl
                               << FG_YELLOW " [SPACE, s]" NO_COLOR " - Save current frame" << std::endl
                               << FG_YELLOW "        [l]" NO_COLOR " - decrease min and max value for IR value range" << std::endl
                               << FG_YELLOW "        [h]" NO_COLOR " - increase min and max value for IR value range" << std::endl
                               << FG_YELLOW "        [1]" NO_COLOR " - decrease min value for IR value range" << std::endl
                               << FG_YELLOW "        [2]" NO_COLOR " - increase min value for IR value range" << std::endl
                               << FG_YELLOW "        [3]" NO_COLOR " - decrease max value for IR value range" << std::endl
                               << FG_YELLOW "        [4]" NO_COLOR " - increase max value for IR value range");

    std::string hints("compressed");
    subImageColor = new image_transport::SubscriberFilter(node.get(), topicColor, hints);
    subImageIr = new image_transport::SubscriberFilter(node.get(), topicIr, hints);
    subImageDepth = new image_transport::SubscriberFilter(node.get(), topicDepth, hints);

    sync = new message_filters::Synchronizer<ColorIrDepthSyncPolicy>(ColorIrDepthSyncPolicy(4), *subImageColor, *subImageIr, *subImageDepth);
    sync->registerCallback(std::bind(&Recorder::callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    OUT_INFO(node, "add node to executor");
    executor->add_node(node);
  }

  void stopRecord()
  {
    executor->remove_node(node);

    delete sync;
    delete subImageColor;
    delete subImageIr;
    delete subImageDepth;
  }

  void convertIr(const cv::Mat &ir, cv::Mat &grey)
  {
    const float factor = 255.0f / (maxIr - minIr);
    grey.create(ir.rows, ir.cols, CV_8U);

#pragma omp parallel for
    for (size_t r = 0; r < (size_t)ir.rows; ++r)
    {
      const uint16_t *itI = ir.ptr<uint16_t>(r);
      uint8_t *itO = grey.ptr<uint8_t>(r);

      for (size_t c = 0; c < (size_t)ir.cols; ++c, ++itI, ++itO)
      {
        *itO = std::min(std::max(*itI - minIr, 0) * factor, 255.0f);
      }
    }
    clahe->apply(grey, grey);
  }

  void findMinMax(const cv::Mat &ir, const std::vector<cv::Point2f> &pointsIr)
  {
    minIr = 0xFFFF;
    maxIr = 0;
    for (size_t i = 0; i < pointsIr.size(); ++i)
    {
      const cv::Point2f &p = pointsIr[i];
      cv::Rect roi(std::max(0, (int)p.x - 2), std::max(0, (int)p.y - 2), 9, 9);
      roi.width = std::min(roi.width, ir.cols - roi.x);
      roi.height = std::min(roi.height, ir.rows - roi.y);

      findMinMax(ir(roi));
    }
  }

  void findMinMax(const cv::Mat &ir)
  {
    for (size_t r = 0; r < (size_t)ir.rows; ++r)
    {
      const uint16_t *it = ir.ptr<uint16_t>(r);

      for (size_t c = 0; c < (size_t)ir.cols; ++c, ++it)
      {
        minIr = std::min(minIr, (int)*it);
        maxIr = std::max(maxIr, (int)*it);
      }
    }
  }

  void callback(const sensor_msgs::msg::Image::ConstSharedPtr imageColor, const sensor_msgs::msg::Image::ConstSharedPtr imageIr, const sensor_msgs::msg::Image::ConstSharedPtr imageDepth)
  {
    std::vector<cv::Point2f> pointsColor, pointsIr;
    cv::Mat color, ir, irGrey, irScaled, depth;
    bool foundColor = false;
    bool foundIr = false;

    if (mode == COLOR || mode == SYNC)
    {
      readImage(imageColor, color);
    }
    if (mode == IR || mode == SYNC)
    {
      readImage(imageIr, ir);
      readImage(imageDepth, depth);
      cv::resize(ir, irScaled, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

      convertIr(irScaled, irGrey);
    }

    if (circleBoard)
    {
      switch (mode)
      {
      case COLOR:
        foundColor = cv::findCirclesGrid(color, boardDims, pointsColor, circleFlags);
        break;
      case IR:
        foundIr = cv::findCirclesGrid(irGrey, boardDims, pointsIr, circleFlags);
        break;
      case SYNC:
        foundColor = cv::findCirclesGrid(color, boardDims, pointsColor, circleFlags);
        foundIr = cv::findCirclesGrid(irGrey, boardDims, pointsIr, circleFlags);
        break;
      }
    }
    else
    {
      const cv::TermCriteria termCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::COUNT, 100, DBL_EPSILON);
      switch (mode)
      {
      case COLOR:
        foundColor = cv::findChessboardCorners(color, boardDims, pointsColor, cv::CALIB_CB_FAST_CHECK);
        break;
      case IR:
        foundIr = cv::findChessboardCorners(irGrey, boardDims, pointsIr, cv::CALIB_CB_ADAPTIVE_THRESH);
        break;
      case SYNC:
        foundColor = cv::findChessboardCorners(color, boardDims, pointsColor, cv::CALIB_CB_FAST_CHECK);
        foundIr = cv::findChessboardCorners(irGrey, boardDims, pointsIr, cv::CALIB_CB_ADAPTIVE_THRESH);
        break;
      }
      if (foundColor)
      {
        cv::cornerSubPix(color, pointsColor, cv::Size(11, 11), cv::Size(-1, -1), termCriteria);
      }
      if (foundIr)
      {
        cv::cornerSubPix(irGrey, pointsIr, cv::Size(11, 11), cv::Size(-1, -1), termCriteria);
      }
    }

    if (foundIr)
    {
      // Update min and max ir value based on checkerboard values
      findMinMax(irScaled, pointsIr);
    }

    lock.lock();
    this->color = color;
    this->ir = ir;
    this->irGrey = irGrey;
    this->depth = depth;
    this->foundColor = foundColor;
    this->foundIr = foundIr;
    this->pointsColor = pointsColor;
    this->pointsIr = pointsIr;
    update = true;
    lock.unlock();
  }

  void display()
  {
    std::vector<cv::Point2f> pointsColor, pointsIr;
    cv::Mat color, ir, irGrey, depth;
    cv::Mat colorDisp, irDisp;
    bool foundColor = false;
    bool foundIr = false;
    bool save = false;
    bool running = true;

    std::chrono::milliseconds duration(1);
    while (!update)
    {
      std::this_thread::sleep_for(duration);
    }

    while (running)
    {
      if (update)
      {
        lock.lock();
        color = this->color;
        ir = this->ir;
        irGrey = this->irGrey;
        depth = this->depth;
        foundColor = this->foundColor;
        foundIr = this->foundIr;
        pointsColor = this->pointsColor;
        pointsIr = this->pointsIr;
        update = false;
        lock.unlock();

        if (mode == COLOR || mode == SYNC)
        {
          cv::cvtColor(color, colorDisp, CV_GRAY2BGR);
          cv::drawChessboardCorners(colorDisp, boardDims, pointsColor, foundColor);
          // cv::resize(colorDisp, colorDisp, cv::Size(), 0.5, 0.5);
          // cv::flip(colorDisp, colorDisp, 1);
        }
        if (mode == IR || mode == SYNC)
        {
          cv::cvtColor(irGrey, irDisp, CV_GRAY2BGR);
          cv::drawChessboardCorners(irDisp, boardDims, pointsIr, foundIr);
          // cv::resize(irDisp, irDisp, cv::Size(), 0.5, 0.5);
          // cv::flip(irDisp, irDisp, 1);
        }
      }

      switch (mode)
      {
      case COLOR:
        cv::imshow("color", colorDisp);
        break;
      case IR:
        cv::imshow("ir", irDisp);
        break;
      case SYNC:
        cv::imshow("color", colorDisp);
        cv::imshow("ir", irDisp);
        break;
      }

      int key = cv::waitKey(10);
      switch (key & 0xFF)
      {
      case ' ':
      case 's':
        OUT_INFO(node, "save image...");
        save = true;
        break;
      case 27:
      case 'q':
        running = false;
        break;
      case '1':
        minIr = std::max(0, minIr - 100);
        break;
      case '2':
        minIr = std::min(maxIr - 1, minIr + 100);
        break;
      case '3':
        maxIr = std::max(minIr + 1, maxIr - 100);
        break;
      case '4':
        maxIr = std::min(0xFFFF, maxIr + 100);
        break;
      case 'l':
        minIr = std::max(0, minIr - 100);
        maxIr = std::max(minIr + 1, maxIr - 100);
        break;
      case 'h':
        maxIr = std::min(0x7FFF, maxIr + 100);
        minIr = std::min(maxIr - 1, minIr + 100);
        break;
      }

      if (save && ((mode == COLOR && foundColor) || (mode == IR && foundIr) || (mode == SYNC && foundColor && foundIr)))
      {
        store(color, ir, irGrey, depth, pointsColor, pointsIr);
        save = false;
      }
    }
    cv::destroyAllWindows();
    cv::waitKey(100);
  }

  void readImage(const sensor_msgs::msg::Image::ConstSharedPtr msgImage, cv::Mat &image) const
  {
    cv_bridge::CvImageConstPtr pCvImage;
    pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
    pCvImage->image.copyTo(image);
  }

  void store(const cv::Mat &color, const cv::Mat &ir, const cv::Mat &irGrey, const cv::Mat &depth, const std::vector<cv::Point2f> &pointsColor, std::vector<cv::Point2f> &pointsIr)
  {
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(4) << frame++;
    const std::string frameNumber(oss.str());
    std::string base = path + frameNumber;

    for (size_t i = 0; i < pointsIr.size(); ++i)
    {
      pointsIr[i].x /= 2.0;
      pointsIr[i].y /= 2.0;
    }

    if (mode == SYNC)
    {
      base += CALIB_SYNC;
    }

    if (mode == COLOR || mode == SYNC)
    {
      cv::imwrite(base + CALIB_FILE_COLOR, color, params);

      cv::FileStorage file(base + CALIB_POINTS_COLOR, cv::FileStorage::WRITE);
      file << "points" << pointsColor;

      OUT_INFO(node, "storing frame: " << frameNumber << " in " << base + CALIB_POINTS_COLOR);
    }

    if (mode == IR || mode == SYNC)
    {
      cv::imwrite(base + CALIB_FILE_IR, ir, params);
      cv::imwrite(base + CALIB_FILE_IR_GREY, irGrey, params);
      cv::imwrite(base + CALIB_FILE_DEPTH, depth, params);

      cv::FileStorage file(base + CALIB_POINTS_IR, cv::FileStorage::WRITE);
      file << "points" << pointsIr;

      OUT_INFO(node, "storing frame: " << frameNumber << " in " << base + CALIB_POINTS_IR);
    }
  }
};

class CameraCalibration
{
private:
  rclcpp::Node::SharedPtr node;

  const bool circleBoard;
  const cv::Size boardDims;
  const float boardSize;
  const int flags;

  const Source mode;
  const std::string path;

  std::vector<cv::Point3f> board;

  std::vector<std::vector<cv::Point3f>> pointsBoard;
  std::vector<std::vector<cv::Point2f>> pointsColor;
  std::vector<std::vector<cv::Point2f>> pointsIr;

  cv::Size sizeColor;
  cv::Size sizeIr;

  cv::Mat cameraMatrixColor, distortionColor, rotationColor, translationColor, projectionColor;
  cv::Mat cameraMatrixIr, distortionIr, rotationIr, translationIr, projectionIr;
  cv::Mat rotation, translation, essential, fundamental, disparity;

  std::vector<cv::Mat> rvecsColor, tvecsColor;
  std::vector<cv::Mat> rvecsIr, tvecsIr;

public:
  CameraCalibration(const rclcpp::Node::SharedPtr &node, const std::string &path, const Source mode, const bool circleBoard, const bool symmetric, const cv::Size &boardDims, const float boardSize, const bool rational)
      : node(node), circleBoard(circleBoard), boardDims(boardDims), boardSize(boardSize), flags(rational ? cv::CALIB_RATIONAL_MODEL : 0), mode(mode), path(path), sizeColor(1920, 1080), sizeIr(512, 424)
  {
    board.resize(boardDims.width * boardDims.height);
    if (symmetric)
    {
      for (size_t r = 0, i = 0; r < (size_t)boardDims.height; ++r)
      {
        for (size_t c = 0; c < (size_t)boardDims.width; ++c, ++i)
        {
          board[i] = cv::Point3f(c * boardSize, r * boardSize, 0);
        }
      }
    }
    else
    {
      for (size_t r = 0, i = 0; r < (size_t)boardDims.height; ++r)
      {
        for (size_t c = 0; c < (size_t)boardDims.width; ++c, ++i)
        {
          board[i] = cv::Point3f(float((2 * c + r % 2) * boardSize), float(r * boardSize), 0); // for asymmetrical circles
        }
      }
    }
  }

  ~CameraCalibration()
  {
  }

  bool restore()
  {
    std::vector<std::string> filesSync;
    std::vector<std::string> filesColor;
    std::vector<std::string> filesIr;

    DIR *dp;
    struct dirent *dirp;
    size_t posColor, posIr, posSync;

    if ((dp = opendir(path.c_str())) == NULL)
    {
      OUT_ERROR(node, "Error opening: " << path);
      return false;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
      std::string filename = dirp->d_name;

      if (dirp->d_type != DT_REG)
      {
        continue;
      }

      posSync = filename.rfind(CALIB_SYNC);
      posColor = filename.rfind(CALIB_FILE_COLOR);

      if (posSync != std::string::npos)
      {
        if (posColor != std::string::npos)
        {
          std::string frameName = filename.substr(0, posColor);
          filesSync.push_back(frameName);
          filesColor.push_back(frameName);
          filesIr.push_back(frameName);
        }
        continue;
      }

      if (posColor != std::string::npos)
      {
        std::string frameName = filename.substr(0, posColor);
        filesColor.push_back(frameName);
        continue;
      }

      posIr = filename.rfind(CALIB_FILE_IR_GREY);
      if (posIr != std::string::npos)
      {
        std::string frameName = filename.substr(0, posIr);
        filesIr.push_back(frameName);
        continue;
      }
    }
    closedir(dp);

    std::sort(filesColor.begin(), filesColor.end());
    std::sort(filesIr.begin(), filesIr.end());
    std::sort(filesSync.begin(), filesSync.end());

    bool ret = true;
    switch (mode)
    {
    case COLOR:
      if (filesColor.empty())
      {
        OUT_ERROR(node, "no files found!");
        return false;
      }
      pointsColor.resize(filesColor.size());
      pointsBoard.resize(filesColor.size(), board);
      ret = ret && readFiles(filesColor, CALIB_POINTS_COLOR, pointsColor);
      break;
    case IR:
      if (filesIr.empty())
      {
        OUT_ERROR(node, "no files found!");
        return false;
      }
      pointsIr.resize(filesIr.size());
      pointsBoard.resize(filesIr.size(), board);
      ret = ret && readFiles(filesIr, CALIB_POINTS_IR, pointsIr);
      break;
    case SYNC:
      if (filesColor.empty() || filesIr.empty())
      {
        OUT_ERROR(node, "no files found!");
        return false;
      }
      pointsColor.resize(filesColor.size());
      pointsIr.resize(filesSync.size());
      pointsColor.resize(filesSync.size());
      pointsBoard.resize(filesSync.size(), board);
      ret = ret && readFiles(filesSync, CALIB_POINTS_COLOR, pointsColor);
      ret = ret && readFiles(filesSync, CALIB_POINTS_IR, pointsIr);
      ret = ret && checkSyncPointsOrder();
      ret = ret && loadCalibration();
      break;
    }
    return ret;
  }

  void calibrate()
  {
    switch (mode)
    {
    case COLOR:
      calibrateIntrinsics(sizeColor, pointsBoard, pointsColor, cameraMatrixColor, distortionColor, rotationColor, projectionColor, rvecsColor, tvecsColor);
      break;
    case IR:
      calibrateIntrinsics(sizeIr, pointsBoard, pointsIr, cameraMatrixIr, distortionIr, rotationIr, projectionIr, rvecsIr, tvecsIr);
      break;
    case SYNC:
      calibrateExtrinsics();
      break;
    }
    storeCalibration();
  }

private:
  bool readFiles(const std::vector<std::string> &files, const std::string &ext, std::vector<std::vector<cv::Point2f>> &points) const
  {
    bool ret = true;
#pragma omp parallel for
    for (size_t i = 0; i < files.size(); ++i)
    {
      std::string pointsname = path + files[i] + ext;

#pragma omp critical
      OUT_INFO(node, "restoring file: " << files[i] << ext);

      cv::FileStorage file(pointsname, cv::FileStorage::READ);
      if (!file.isOpened())
      {
#pragma omp critical
        {
          ret = false;
          OUT_ERROR(node, "couldn't open file: " << files[i] << ext);
        }
      }
      else
      {
        file["points"] >> points[i];
      }
    }
    return ret;
  }

  bool checkSyncPointsOrder()
  {
    if (pointsColor.size() != pointsIr.size())
    {
      OUT_ERROR(node, "number of detected color and ir patterns does not match!");
      return false;
    }

    for (size_t i = 0; i < pointsColor.size(); ++i)
    {
      const std::vector<cv::Point2f> &pColor = pointsColor[i];
      const std::vector<cv::Point2f> &pIr = pointsIr[i];

      if (pColor.front().y > pColor.back().y || pColor.front().x > pColor.back().x)
      {
        std::reverse(pointsColor[i].begin(), pointsColor[i].end());
      }

      if (pIr.front().y > pIr.back().y || pIr.front().x > pIr.back().x)
      {
        std::reverse(pointsIr[i].begin(), pointsIr[i].end());
      }
    }
    return true;
  }

  void calibrateIntrinsics(const cv::Size &size, const std::vector<std::vector<cv::Point3f>> &pointsBoard, const std::vector<std::vector<cv::Point2f>> &points,
                           cv::Mat &cameraMatrix, cv::Mat &distortion, cv::Mat &rotation, cv::Mat &projection, std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs)
  {
    if (points.empty())
    {
      OUT_ERROR(node, "no data for calibration provided!");
      return;
    }
    const cv::TermCriteria termCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, DBL_EPSILON);
    double error;

    OUT_INFO(node, "calibrating intrinsics...");
    error = cv::calibrateCamera(pointsBoard, points, size, cameraMatrix, distortion, rvecs, tvecs, flags, termCriteria);
    OUT_INFO(node, "re-projection error: " << error << std::endl);

    OUT_INFO(node, "Camera Matrix:" << std::endl
                                    << cameraMatrix);
    OUT_INFO(node, "Distortion Coeeficients:" << std::endl
                                              << distortion << std::endl);
    rotation = cv::Mat::eye(3, 3, CV_64F);
    projection = cv::Mat::eye(4, 4, CV_64F);
    cameraMatrix.copyTo(projection(cv::Rect(0, 0, 3, 3)));
  }

  void calibrateExtrinsics()
  {
    if (pointsColor.size() != pointsIr.size())
    {
      OUT_ERROR(node, "number of detected color and ir patterns does not match!");
      return;
    }
    if (pointsColor.empty() || pointsIr.empty())
    {
      OUT_ERROR(node, "no data for calibration provided!");
      return;
    }
    const cv::TermCriteria termCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON);
    double error;

    OUT_INFO(node, "Camera Matrix Color:" << std::endl
                                          << cameraMatrixColor);
    OUT_INFO(node, "Distortion Coeeficients Color:" << std::endl
                                                    << distortionColor << std::endl);
    OUT_INFO(node, "Camera Matrix Ir:" << std::endl
                                       << cameraMatrixIr);
    OUT_INFO(node, "Distortion Coeeficients Ir:" << std::endl
                                                 << distortionIr << std::endl);

    OUT_INFO(node, "calibrating Color and Ir extrinsics...");
#if CV_MAJOR_VERSION == 2
    error = cv::stereoCalibrate(pointsBoard, pointsIr, pointsColor, cameraMatrixIr, distortionIr, cameraMatrixColor, distortionColor, sizeColor,
                                rotation, translation, essential, fundamental, termCriteria, cv::CALIB_FIX_INTRINSIC);
#elif CV_MAJOR_VERSION == 3 || CV_MAJOR_VERSION == 4
    error = cv::stereoCalibrate(pointsBoard, pointsIr, pointsColor, cameraMatrixIr, distortionIr, cameraMatrixColor, distortionColor, sizeColor,
                                rotation, translation, essential, fundamental, cv::CALIB_FIX_INTRINSIC, termCriteria);
#endif
    OUT_INFO(node, "re-projection error: " << error << std::endl);

    OUT_INFO(node, "Rotation:" << std::endl
                               << rotation);
    OUT_INFO(node, "Translation:" << std::endl
                                  << translation);
    OUT_INFO(node, "Essential:" << std::endl
                                << essential);
    OUT_INFO(node, "Fundamental:" << std::endl
                                  << fundamental << std::endl);
  }

  void storeCalibration()
  {
    cv::FileStorage fs;

    switch (mode)
    {
    case SYNC:
      fs.open(path + K2_CALIB_POSE, cv::FileStorage::WRITE);
      break;
    case COLOR:
      fs.open(path + K2_CALIB_COLOR, cv::FileStorage::WRITE);
      break;
    case IR:
      fs.open(path + K2_CALIB_IR, cv::FileStorage::WRITE);
      break;
    }

    if (!fs.isOpened())
    {
      OUT_ERROR(node, "couldn't store calibration data!");
      return;
    }

    switch (mode)
    {
    case SYNC:
      fs << K2_CALIB_ROTATION << rotation;
      fs << K2_CALIB_TRANSLATION << translation;
      fs << K2_CALIB_ESSENTIAL << essential;
      fs << K2_CALIB_FUNDAMENTAL << fundamental;
      break;
    case COLOR:
      fs << K2_CALIB_CAMERA_MATRIX << cameraMatrixColor;
      fs << K2_CALIB_DISTORTION << distortionColor;
      fs << K2_CALIB_ROTATION << rotationColor;
      fs << K2_CALIB_PROJECTION << projectionColor;
      break;
    case IR:
      fs << K2_CALIB_CAMERA_MATRIX << cameraMatrixIr;
      fs << K2_CALIB_DISTORTION << distortionIr;
      fs << K2_CALIB_ROTATION << rotationIr;
      fs << K2_CALIB_PROJECTION << projectionIr;
      break;
    }
    fs.release();
  }

  bool loadCalibration()
  {
    cv::FileStorage fs;

    if (fs.open(path + K2_CALIB_COLOR, cv::FileStorage::READ))
    {
      fs[K2_CALIB_CAMERA_MATRIX] >> cameraMatrixColor;
      fs[K2_CALIB_DISTORTION] >> distortionColor;
      fs[K2_CALIB_ROTATION] >> rotationColor;
      fs[K2_CALIB_PROJECTION] >> projectionColor;
      fs.release();
    }
    else
    {
      OUT_ERROR(node, "couldn't load color calibration data!");
      return false;
    }

    if (fs.open(path + K2_CALIB_IR, cv::FileStorage::READ))
    {
      fs[K2_CALIB_CAMERA_MATRIX] >> cameraMatrixIr;
      fs[K2_CALIB_DISTORTION] >> distortionIr;
      fs[K2_CALIB_ROTATION] >> rotationIr;
      fs[K2_CALIB_PROJECTION] >> projectionIr;
      fs.release();
    }
    else
    {
      OUT_ERROR(node, "couldn't load ir calibration data!");
      return false;
    }

    return true;
  }
};

class DepthCalibration
{
private:
  rclcpp::Node::SharedPtr node;

  const std::string path;

  std::vector<cv::Point3f> board;
  std::vector<std::vector<cv::Point2f>> points;
  std::vector<std::string> images;

  cv::Size size;

  cv::Mat cameraMatrix, distortion, rotation, translation;
  cv::Mat mapX, mapY;

  double fx, fy, cx, cy;

  std::ofstream plot;

public:
  DepthCalibration(const rclcpp::Node::SharedPtr &node, const std::string &path, const bool symmetric, const cv::Size &boardDims, const float boardSize)
      : node(node), path(path), size(512, 424)
  {
    board.resize(boardDims.width * boardDims.height);
    if (symmetric)
    {
      for (size_t r = 0, i = 0; r < (size_t)boardDims.height; ++r)
      {
        for (size_t c = 0; c < (size_t)boardDims.width; ++c, ++i)
        {
          board[i] = cv::Point3f(c * boardSize, r * boardSize, 0);
        }
      }
    }
    else
    {
      for (size_t r = 0, i = 0; r < (size_t)boardDims.height; ++r)
      {
        for (size_t c = 0; c < (size_t)boardDims.width; ++c, ++i)
        {
          board[i] = cv::Point3f(float((2 * c + r % 2) * boardSize), float(r * boardSize), 0); // for asymmetrical circles
        }
      }
    }
  }

  ~DepthCalibration()
  {
  }

  bool restore()
  {
    std::vector<std::string> files;

    DIR *dp;
    struct dirent *dirp;
    size_t pos;

    if ((dp = opendir(path.c_str())) == NULL)
    {
      OUT_ERROR(node, "Error opening: " << path);
      return false;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
      std::string filename = dirp->d_name;

      if (dirp->d_type != DT_REG)
      {
        continue;
      }

      /*pos = filename.rfind(CALIB_SYNC);
      if(pos != std::string::npos)
      {
        continue;
      }*/

      pos = filename.rfind(CALIB_FILE_IR_GREY);
      if (pos != std::string::npos)
      {
        std::string frameName = filename.substr(0, pos);
        files.push_back(frameName);
        continue;
      }
    }
    closedir(dp);

    std::sort(files.begin(), files.end());

    if (files.empty())
    {
      OUT_ERROR(node, "no files found!");
      return false;
    }

    bool ret = readFiles(files);
    ret = ret && loadCalibration();

    if (ret)
    {
      cv::initUndistortRectifyMap(cameraMatrix, distortion, cv::Mat(), cameraMatrix, size, CV_32FC1, mapX, mapY);
      fx = cameraMatrix.at<double>(0, 0);
      fy = cameraMatrix.at<double>(1, 1);
      cx = cameraMatrix.at<double>(0, 2);
      cy = cameraMatrix.at<double>(1, 2);
    }
    return ret;
  }

  void calibrate()
  {
    plot.open(path + "plot.dat", std::ios_base::trunc);
    if (!plot.is_open())
    {
      OUT_ERROR(node, "couldn't open 'plot.dat'!");
      return;
    }
    if (images.empty())
    {
      OUT_ERROR(node, "no images found!");
      return;
    }

    plot << "# Columns:" << std::endl
         << "# 1: X" << std::endl
         << "# 2: Y" << std::endl
         << "# 3: computed depth" << std::endl
         << "# 4: measured depth" << std::endl
         << "# 5: difference between computed and measured depth" << std::endl;

    std::vector<double> depthDists, imageDists;
    for (size_t i = 0; i < images.size(); ++i)
    {
      OUT_INFO(node, "frame: " << images[i]);
      plot << "# frame: " << images[i] << std::endl;

      cv::Mat depth, planeNormal, region;
      double planeDistance;
      cv::Rect roi;

      depth = cv::imread(images[i], cv::IMREAD_ANYDEPTH);
      if (depth.empty())
      {
        OUT_ERROR(node, "couldn't load image '" << images[i] << "'!");
        return;
      }

      cv::remap(depth, depth, mapX, mapY, cv::INTER_NEAREST);
      computeROI(depth, points[i], region, roi);

      getPlane(i, planeNormal, planeDistance);

      computePointDists(planeNormal, planeDistance, region, roi, depthDists, imageDists);
    }
    compareDists(imageDists, depthDists);
  }

private:
  void compareDists(const std::vector<double> &imageDists, const std::vector<double> &depthDists) const
  {
    if (imageDists.size() != depthDists.size())
    {
      OUT_ERROR(node, "number of real and computed distance samples does not match!");
      return;
    }
    if (imageDists.empty() || depthDists.empty())
    {
      OUT_ERROR(node, "no distance sample data!");
      return;
    }

    double avg = 0, sqavg = 0, var = 0, stddev = 0;
    std::vector<double> diffs(imageDists.size());

    for (size_t i = 0; i < imageDists.size(); ++i)
    {
      diffs[i] = imageDists[i] - depthDists[i];
      avg += diffs[i];
      sqavg += diffs[i] * diffs[i];
    }
    sqavg = sqrt(sqavg / imageDists.size());
    avg /= imageDists.size();

    for (size_t i = 0; i < imageDists.size(); ++i)
    {
      const double diff = diffs[i] - avg;
      var += diff * diff;
    }
    var = var / (imageDists.size());
    stddev = sqrt(var);

    std::sort(diffs.begin(), diffs.end());
    OUT_INFO(node, "stats on difference:" << std::endl
                                          << "     avg: " << avg << std::endl
                                          << "     var: " << var << std::endl
                                          << "  stddev: " << stddev << std::endl
                                          << "     rms: " << sqavg << std::endl
                                          << "  median: " << diffs[diffs.size() / 2]);

    storeCalibration(avg * 1000.0);
  }

  void computePointDists(const cv::Mat &normal, const double distance, const cv::Mat &region, const cv::Rect &roi, std::vector<double> &depthDists, std::vector<double> &imageDists)
  {
    for (int r = 0; r < region.rows; ++r)
    {
      const uint16_t *itD = region.ptr<uint16_t>(r);
      cv::Point p(roi.x, roi.y + r);

      for (int c = 0; c < region.cols; ++c, ++itD, ++p.x)
      {
        const double dDist = *itD / 1000.0;

        if (dDist < 0.1)
        {
          continue;
        }

        const double iDist = computeDistance(p, normal, distance);
        const double diff = iDist - dDist;

        if (std::abs(diff) > 0.08)
        {
          continue;
        }
        depthDists.push_back(dDist);
        imageDists.push_back(iDist);
        plot << p.x << ' ' << p.y << ' ' << iDist << ' ' << dDist << ' ' << diff << std::endl;
      }
    }
  }

  double computeDistance(const cv::Point &pointImage, const cv::Mat &normal, const double distance) const
  {
    cv::Mat point = cv::Mat(3, 1, CV_64F);

    point.at<double>(0) = (pointImage.x - cx) / fx;
    point.at<double>(1) = (pointImage.y - cy) / fy;
    point.at<double>(2) = 1;

    double t = distance / normal.dot(point);
    point = point * t;

    return point.at<double>(2);
  }

  void getPlane(const size_t index, cv::Mat &normal, double &distance) const
  {
    cv::Mat rvec, rotation, translation;

    cv::solvePnPRansac(board, points[index], cameraMatrix, distortion, rvec, translation, false, 300, 0.05, 0.99, cv::noArray(), cv::SOLVEPNP_ITERATIVE);

    cv::Rodrigues(rvec, rotation);

    normal = cv::Mat(3, 1, CV_64F);
    normal.at<double>(0) = 0;
    normal.at<double>(1) = 0;
    normal.at<double>(2) = 1;
    normal = rotation * normal;
    distance = normal.dot(translation);
  }

  void computeROI(const cv::Mat &depth, const std::vector<cv::Point2f> &points, cv::Mat &region, cv::Rect &roi) const
  {
    std::vector<cv::Point2f> norm;
    std::vector<cv::Point> undist, hull;

    cv::undistortPoints(points, norm, cameraMatrix, distortion);
    undist.reserve(norm.size());

    for (size_t i = 0; i < norm.size(); ++i)
    {
      cv::Point p;
      p.x = (int)round(norm[i].x * fx + cx);
      p.y = (int)round(norm[i].y * fy + cy);
      if (p.x >= 0 && p.x < depth.cols && p.y >= 0 && p.y < depth.rows)
      {
        undist.push_back(p);
      }
    }

    roi = cv::boundingRect(undist);

    cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8U);

    cv::convexHull(undist, hull);
    cv::fillConvexPoly(mask, hull, CV_RGB(255, 255, 255));

    cv::Mat tmp;
    depth.copyTo(tmp, mask);
    tmp(roi).copyTo(region);
  }

  bool readFiles(const std::vector<std::string> &files)
  {
    points.resize(files.size());
    images.resize(files.size());
    bool ret = true;

#pragma omp parallel for
    for (size_t i = 0; i < files.size(); ++i)
    {
      std::string pointsname = path + files[i] + CALIB_POINTS_IR;

#pragma omp critical
      OUT_INFO(node, "restoring file: " << files[i]);

      cv::FileStorage file(pointsname, cv::FileStorage::READ);
      if (!file.isOpened())
      {
#pragma omp critical
        {
          OUT_ERROR(node, "couldn't read '" << pointsname << "'!");
          ret = false;
        }
      }
      else
      {
        file["points"] >> points[i];
        file.release();
        images[i] = path + files[i] + CALIB_FILE_DEPTH;
      }
    }
    return ret;
  }

  bool loadCalibration()
  {
    cv::FileStorage fs;

    if (fs.open(path + K2_CALIB_IR, cv::FileStorage::READ))
    {
      fs[K2_CALIB_CAMERA_MATRIX] >> cameraMatrix;
      fs[K2_CALIB_DISTORTION] >> distortion;
      fs.release();
    }
    else
    {
      OUT_ERROR(node, "couldn't read calibration '" << path + K2_CALIB_IR << "'!");
      return false;
    }

    return true;
  }

  void storeCalibration(const double depthShift) const
  {
    cv::FileStorage fs;

    if (fs.open(path + K2_CALIB_DEPTH, cv::FileStorage::WRITE))
    {
      fs << K2_CALIB_DEPTH_SHIFT << depthShift;
      fs.release();
    }
    else
    {
      OUT_ERROR(node, "couldn't store depth calibration!");
    }
  }
};

void help(const std::string &path)
{
  std::cout << path << FG_BLUE " [options]" << std::endl
            << FG_GREEN "  name" NO_COLOR ": " FG_YELLOW "'any string'" NO_COLOR " equals to the kinect2_bridge topic base name" << std::endl
            << FG_GREEN "  mode" NO_COLOR ": " FG_YELLOW "'record'" NO_COLOR " or " FG_YELLOW "'calibrate'" << std::endl
            << FG_GREEN "  source" NO_COLOR ": " FG_YELLOW "'color'" NO_COLOR ", " FG_YELLOW "'ir'" NO_COLOR ", " FG_YELLOW "'sync'" NO_COLOR ", " FG_YELLOW "'depth'" << std::endl
            << FG_GREEN "  board" NO_COLOR ":" << std::endl
            << FG_YELLOW "    'circle<WIDTH>x<HEIGHT>x<SIZE>'  " NO_COLOR "for symmetric circle grid" << std::endl
            << FG_YELLOW "    'acircle<WIDTH>x<HEIGHT>x<SIZE>' " NO_COLOR "for asymmetric circle grid" << std::endl
            << FG_YELLOW "    'chess<WIDTH>x<HEIGHT>x<SIZE>'   " NO_COLOR "for chessboard pattern" << std::endl
            << FG_GREEN "  distortion model" NO_COLOR ": " FG_YELLOW "'rational'" NO_COLOR " for using model with 8 instead of 5 coefficients" << std::endl
            << FG_GREEN "  output path" NO_COLOR ": " FG_YELLOW "'-path <PATH>'" NO_COLOR << std::endl;
}

int main(int argc, char **argv)
{
#if EXTENDED_OUTPUT
  ROSCONSOLE_AUTOINIT;
  if (!getenv("ROSCONSOLE_FORMAT"))
  {
    ros::console::g_formatter.tokens_.clear();
    ros::console::g_formatter.init("[${severity}] ${message}");
  }
#endif

  Mode mode = RECORD;
  Source source = SYNC;
  bool circleBoard = false;
  bool symmetric = true;
  bool rational = false;
  bool calibDepth = false;
  cv::Size boardDims = cv::Size(7, 5);
  float boardSize = 0.108;
  std::string ns = K2_DEFAULT_NS;
  std::string path = "./";

  bool running = true;

  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("kinect2_calib");
  auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();

  auto exec_thread_handle = std::thread([&]()
                                        { OUT_INFO(node, "spinning...");
                                          while(running) executor->spin_once(std::chrono::nanoseconds(1000)); });

  for (int argI = 1; argI < argc; ++argI)
  {
    std::string arg(argv[argI]);

    if (arg == "--help" || arg == "--h" || arg == "-h" || arg == "-?" || arg == "--?")
    {
      help(argv[0]);
      rclcpp::shutdown();
      return 0;
    }
    else if (arg == "record")
    {
      mode = RECORD;
    }
    else if (arg == "calibrate")
    {
      mode = CALIBRATE;
    }
    else if (arg == "color")
    {
      source = COLOR;
    }
    else if (arg == "ir")
    {
      source = IR;
    }
    else if (arg == "sync")
    {
      source = SYNC;
    }
    else if (arg == "depth")
    {
      calibDepth = true;
    }
    else if (arg == "rational")
    {
      rational = true;
    }
    else if (arg.find("circle") == 0 && arg.find('x') != arg.rfind('x') && arg.rfind('x') != std::string::npos)
    {
      circleBoard = true;
      const size_t start = 6;
      const size_t leftX = arg.find('x');
      const size_t rightX = arg.rfind('x');
      const size_t end = arg.size();

      int width = atoi(arg.substr(start, leftX - start).c_str());
      int height = atoi(arg.substr(leftX + 1, rightX - leftX + 1).c_str());
      boardSize = atof(arg.substr(rightX + 1, end - rightX + 1).c_str());
      boardDims = cv::Size(width, height);
    }
    else if ((arg.find("circle") == 0 || arg.find("acircle") == 0) && arg.find('x') != arg.rfind('x') && arg.rfind('x') != std::string::npos)
    {
      symmetric = arg.find("circle") == 0;
      circleBoard = true;
      const size_t start = 6 + (symmetric ? 0 : 1);
      const size_t leftX = arg.find('x');
      const size_t rightX = arg.rfind('x');
      const size_t end = arg.size();

      int width = atoi(arg.substr(start, leftX - start).c_str());
      int height = atoi(arg.substr(leftX + 1, rightX - leftX + 1).c_str());
      boardSize = atof(arg.substr(rightX + 1, end - rightX + 1).c_str());
      boardDims = cv::Size(width, height);
    }
    else if (arg.find("chess") == 0 && arg.find('x') != arg.rfind('x') && arg.rfind('x') != std::string::npos)
    {
      circleBoard = false;
      const size_t start = 5;
      const size_t leftX = arg.find('x');
      const size_t rightX = arg.rfind('x');
      const size_t end = arg.size();

      int width = atoi(arg.substr(start, leftX - start).c_str());
      int height = atoi(arg.substr(leftX + 1, rightX - leftX + 1).c_str());
      boardSize = atof(arg.substr(rightX + 1, end - rightX + 1).c_str());
      boardDims = cv::Size(width, height);
    }
    else if (arg == "-path" && ++argI < argc)
    {
      arg = argv[argI];
      struct stat fileStat;
      if (stat(arg.c_str(), &fileStat) == 0 && S_ISDIR(fileStat.st_mode))
      {
        path = arg;
      }
      else
      {
        OUT_ERROR(node, "Unknown path: " << arg);
        help(argv[0]);
        rclcpp::shutdown();
        return 0;
      }
    }
    else
    {
      ns = arg;
    }
  }

  std::string topicColor = "/" + ns + K2_TOPIC_HD + K2_TOPIC_IMAGE_MONO;
  std::string topicIr = "/" + ns + K2_TOPIC_SD + K2_TOPIC_IMAGE_IR;
  std::string topicDepth = "/" + ns + K2_TOPIC_SD + K2_TOPIC_IMAGE_DEPTH;
  OUT_INFO(node, "Start settings:" << std::endl
                                   << "       Mode: " FG_CYAN << (mode == RECORD ? "record" : "calibrate") << NO_COLOR << std::endl
                                   << "     Source: " FG_CYAN << (calibDepth ? "depth" : (source == COLOR ? "color" : (source == IR ? "ir" : "sync"))) << NO_COLOR << std::endl
                                   << "      Board: " FG_CYAN << (circleBoard ? "circles" : "chess") << NO_COLOR << std::endl
                                   << " Dimensions: " FG_CYAN << boardDims.width << " x " << boardDims.height << NO_COLOR << std::endl
                                   << " Field size: " FG_CYAN << boardSize << NO_COLOR << std::endl
                                   << "Dist. model: " FG_CYAN << (rational ? '8' : '5') << " coefficients" << NO_COLOR << std::endl
                                   << "Topic color: " FG_CYAN << topicColor << NO_COLOR << std::endl
                                   << "   Topic ir: " FG_CYAN << topicIr << NO_COLOR << std::endl
                                   << "Topic depth: " FG_CYAN << topicDepth << NO_COLOR << std::endl
                                   << "       Path: " FG_CYAN << path << NO_COLOR << std::endl);

  if (mode == RECORD)
  {
    Recorder recorder(node, executor, path, topicColor, topicIr, topicDepth, source, circleBoard, symmetric, boardDims, boardSize);

    OUT_INFO(node, "starting recorder...");
    recorder.run();

    OUT_INFO(node, "stopped recording...");
  }
  else if (calibDepth)
  {
    DepthCalibration calib(node, path, symmetric, boardDims, boardSize);

    OUT_INFO(node, "restoring files...");
    calib.restore();

    OUT_INFO(node, "starting calibration...");
    calib.calibrate();
  }
  else
  {
    CameraCalibration calib(node, path, source, circleBoard, symmetric, boardDims, boardSize, rational);

    OUT_INFO(node, "restoring files...");
    calib.restore();

    OUT_INFO(node, "starting calibration...");
    calib.calibrate();
  }

  running = false;
  exec_thread_handle.join();

  rclcpp::shutdown();

  return 0;
}
